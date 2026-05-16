import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tabm
from utils import seed_everything, get_cluster_acc, datasets_to_c
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def _parse_args(args):
    parser = argparse.ArgumentParser()
    #dataset
    parser.add_argument('--dataset', type=str, default="CIFAR10", help="Dataset to run TURTLE")
    parser.add_argument('--phis', type=str, default=["clipvitB32"], nargs='+',
                        help="Representation spaces to run TURTLE")
    # training - 进一步优化参数
    parser.add_argument('--gamma', type=float, default=50.0, help='降低熵正则化强度')
    parser.add_argument('--T', type=int, default=6000, help='减少总迭代次数')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='进一步降低内循环学习率')
    parser.add_argument('--outer_lr', type=float, default=0.01, help='进一步降低外循环学习率')
    parser.add_argument('--batch_size', type=int, default=1000, help='继续减小批次大小')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--M', type=int, default=10, help='大幅减少内循环步数')
    # others
    parser.add_argument('--cross_val', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--root_dir', type=str, default="data")
    parser.add_argument('--seed', type=int, default=66)

    # batchensemble - 简化模型
    parser.add_argument('--k', type=int, default=32, help='大幅减少集成成员数量')
    parser.add_argument('--d_in', type=int, default=24)
    parser.add_argument('--d', type=int, default=512, help='大幅减小隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--tabm_init', action='store_true')
    parser.add_argument('--scaling_init', type=str, default='normal')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--topk', type=int, default=3, help='大幅减少邻居数量')

    # 稳定性参数
    parser.add_argument('--grad_clip', type=float, default=1.0, help='更严格的梯度裁剪')
    parser.add_argument('--eps', type=float, default=1e-6, help='数值稳定性常数')
    parser.add_argument('--warmup_steps', type=int, default=200, help='预热步数')
    parser.add_argument('--eval_freq', type=int, default=50, help='评估频率')
    return parser.parse_args(args)

def safe_softmax(logits, dim=-1, eps=1e-6):
    """数值稳定的softmax，修复math未定义问题"""
    # 移除math依赖，使用纯PyTorch实现
    logits = torch.clamp(logits, -50, 50)  # 限制logits范围防止溢出
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits)
    return exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + eps)


def safe_log(x, eps=1e-6):
    """数值稳定的log函数"""
    return torch.log(torch.clamp(x, min=eps))


class DistillLoss(nn.Module):
    """蒸馏损失函数，与ensemble_en.py中保持一致"""

    def __init__(self, temperature=2.0):
        super(DistillLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, neighbor_logits):
        # 添加数值稳定性检查
        logits = torch.clamp(logits, -50, 50)
        neighbor_logits = torch.clamp(neighbor_logits, -50, 50)

        # 计算KL散度作为蒸馏损失，与ensemble_en.py中保持一致
        loss = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1),
            F.softmax(neighbor_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        return loss


def consistency_loss(anchors, neighbors):
    # 添加数值稳定性处理
    b, n = anchors.size()
    # 使用 cosine similarity 替代点积，避免数值过大
    similarity = F.cosine_similarity(anchors, neighbors, dim=1)
    # 将相似度限制在 [0, 1] 范围内
    similarity = torch.clamp(similarity, min=0.0, max=1.0)
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones)

    return consistency_loss


def entropy(logit_):
    """计算熵，添加数值稳定性处理"""
    # 使用 torch.special.entr 函数，它能更好地处理数值稳定性
    logit_ = torch.clamp(logit_, -50, 50)  # 限制输入范围
    probs = F.softmax(logit_, dim=-1)
    # torch.special.entr(x) = -x * log(x)，且能正确处理 x=0 的情况
    entropy_values = torch.special.entr(probs)
    return entropy_values.sum(dim=-1).mean()


def mine_nearest_neighbors(features, topk=10):
    """
    计算最近邻索引，与ensemble_en.py中保持一致
    """
    n_samples = features.shape[0]

    if n_samples > 10000:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=topk + 1, algorithm='brute', metric='cosine').fit(features)
        _, indices = nbrs.kneighbors(features)
        return indices[:, 1:]
    else:
        similarities = np.dot(features, features.T)
        indices = np.argpartition(similarities, -topk - 1, axis=1)[:, -topk - 1:]
        return indices[:, :-1]


def robust_data_validation(Zs_train, Zs_val):
    """增强的数据验证和预处理，修复inf标准差问题"""
    Zs_train_processed = []
    Zs_val_processed = []

    for i, (Z_train, Z_val) in enumerate(zip(Zs_train, Zs_val)):
        # print(f"处理特征空间 {i + 1}: 原始形状 {Z_train.shape}")

        # 检查并修复NaN和Inf值
        if np.isnan(Z_train).any():
            # print(f"特征空间 {i + 1}: 发现NaN值，进行清理")
            Z_train = np.nan_to_num(Z_train, nan=0.0)
        if np.isinf(Z_train).any():
            # print(f"特征空间 {i + 1}: 发现Inf值，进行清理")
            Z_train = np.nan_to_num(Z_train, posinf=1.0, neginf=-1.0)

        # 检查数据范围并修复极端值
        data_min, data_max = Z_train.min(), Z_train.max()
        if abs(data_min) > 100 or abs(data_max) > 100:
            # print(f"特征空间 {i + 1}: 检测到极端值范围 [{data_min:.3f}, {data_max:.3f}]，进行裁剪")
            Z_train = np.clip(Z_train, -10.0, 10.0)
            Z_val = np.clip(Z_val, -10.0, 10.0)

        # 稳健的标准化（处理标准差为0、inf或极小值的情况）
        mean = Z_train.mean(axis=0)
        std = Z_train.std(axis=0)

        # 修复标准差问题：同时处理inf、0和极小值
        # 将inf、0或极小标准差（<1e-8）替换为1.0
        std = np.where((np.isinf(std)) | (std == 0) | (std < 1e-8), 1.0, std)

        Z_train_norm = (Z_train - mean) / std
        Z_val_norm = (Z_val - mean) / std

        # 再次检查标准化后的范围并处理可能的NaN/Inf值
        if np.isnan(Z_train_norm).any() or np.isinf(Z_train_norm).any():
            # print(f"特征空间 {i + 1}: 标准化后仍有NaN/Inf值，进行最终清理")
            Z_train_norm = np.nan_to_num(Z_train_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.isnan(Z_val_norm).any() or np.isinf(Z_val_norm).any():
            # print(f"特征空间 {i + 1}: 验证集标准化后仍有NaN/Inf值，进行最终清理")
            Z_val_norm = np.nan_to_num(Z_val_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        # 检查标准化后的范围
        train_min, train_max = Z_train_norm.min(), Z_train_norm.max()
        print(f"特征空间 {i + 1}: 标准化后范围 [{train_min:.3f}, {train_max:.3f}]")

        Zs_train_processed.append(Z_train_norm.astype(np.float32))
        Zs_val_processed.append(Z_val_norm.astype(np.float32))

    return Zs_train_processed, Zs_val_processed


def create_simple_inner_classifier(input_dim, output_dim, args):
    """创建简化的内部分类器，采用BatchEnsemble集成方式"""

    class SimpleMultiModalClassifier(nn.Module):
        def __init__(self, input_dim, output_dim, args):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.args = args

            # 安全计算模态维度
            self.modal_dim = max(1, input_dim // 2)  # 确保至少为1

            # print(f"创建分类器: 输入维度={input_dim}, 模态维度={self.modal_dim}, 输出维度={output_dim}")

            # 图像模态处理网络 - 使用BatchEnsemble结构
            self.image_branch = nn.Sequential(
                # Ensemble view layer
                tabm.EnsembleView(k=args.k),

                # First LinearBatchEnsemble layer
                tabm.LinearBatchEnsemble(
                    self.modal_dim, output_dim, k=args.k,
                    scaling_init=(args.scaling_init, 'ones') if args.tabm_init else args.scaling_init,
                )

            )

            # 文本模态处理网络 - 使用BatchEnsemble结构
            self.text_branch = nn.Sequential(
                # Ensemble view layer
                tabm.EnsembleView(k=args.k),

                # First LinearBatchEnsemble layer
                tabm.LinearBatchEnsemble(
                    self.modal_dim, output_dim, k=args.k,
                    scaling_init=(args.scaling_init, 'ones') if args.tabm_init else args.scaling_init,
                )
            )

            # 蒸馏损失函数，与ensemble_en.py中保持一致
            self.distill_loss = DistillLoss(temperature=args.temperature)

            # 初始化权重
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, x, neighbor_image_x=None, neighbor_text_x=None):
            # 基础验证
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("警告: 输入包含NaN或Inf值")
                x = torch.clamp(x, min=-10.0, max=10.0)

            # 安全分离特征
            image_features = x[:, :self.modal_dim]
            text_features = x[:, self.modal_dim:]

            image_output = self.image_branch(image_features)
            text_output = self.text_branch(text_features)

            loss = torch.tensor(0.0, device=x.device)
            loss_details = {'distill_loss': 0.0, 'consist_loss': 0.0, 'entropy_loss': 0.0}

            # 与ensemble_en.py中保持一致的损失计算
            if neighbor_image_x is not None and neighbor_text_x is not None:
                try:
                    # 获取邻居的模态特征
                    neighbor_image_features = neighbor_image_x[:, :self.modal_dim]
                    neighbor_text_features = neighbor_text_x[:, self.modal_dim:]  # 文本邻居的文本部分

                    # 获取邻居的模态输出
                    neighbor_image_output = self.image_branch(neighbor_image_features)
                    neighbor_text_output = self.text_branch(neighbor_text_features)

                    # 修改：先对所有ensemble成员求平均，然后再计算损失
                    # 对每个模态的输出求平均
                    image_output_mean = image_output.mean(dim=1)  # [batch_size, output_dim]
                    text_output_mean = text_output.mean(dim=1)    # [batch_size, output_dim]
                    neighbor_image_output_mean = neighbor_image_output.mean(dim=1)  # [batch_size, output_dim]
                    neighbor_text_output_mean = neighbor_text_output.mean(dim=1)    # [batch_size, output_dim]

                    # 计算蒸馏损失（使用平均后的输出）
                    distill_loss_img_to_text = self.distill_loss(image_output_mean, neighbor_text_output_mean)
                    distill_loss_text_to_img = self.distill_loss(text_output_mean, neighbor_image_output_mean)

                    # 两种蒸馏损失的平均
                    distill_loss = distill_loss_img_to_text + distill_loss_text_to_img

                    # 计算一致性损失（使用平均后的输出）
                    consist_loss = consistency_loss(image_output_mean, text_output_mean)

                    # 计算熵损失（使用平均后的输出）
                    entropy_loss = entropy(image_output_mean) + entropy(text_output_mean)

                    # 总损失
                    loss = 0.1 * distill_loss + 0.1 * consist_loss - 0.01 * entropy_loss
                    # loss = distill_loss
                    loss_details['distill_loss'] = distill_loss.item()
                    loss_details['consist_loss'] = consist_loss.item()
                    loss_details['entropy_loss'] = entropy_loss.item()
                except Exception as e:
                    print(f"损失计算错误: {e}")
                    loss = torch.tensor(0.0, device=x.device)

            # 组合输出 - 对每个ensemble成员分别计算后求平均
            combined_output =(image_output+text_output)/2
            # combined_output =image_output
            return combined_output, loss, loss_details

    return SimpleMultiModalClassifier(input_dim, output_dim, args).to(args.device)


class SimpleLRScheduler:
    """简化的学习率调度器，避免复杂依赖"""

    def __init__(self, optimizer, total_steps, init_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        # 线性衰减
        progress = self.current_step / self.total_steps
        lr = self.init_lr * (1 - progress) + self.min_lr * progress

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def has_invalid_gradients(model):
    """检查模型是否有无效梯度"""
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return True
            # 检查梯度是否过大
            if param.grad.abs().max() > 1000:
                return True
    return False


def run_stable_fixed(args=None):
    """完全修复的主训练函数"""
    args = _parse_args(args)
    seed_everything(args.seed)

    print("=== 完全修复训练启动 ===")
    print(f"设备: {args.device}")
    print(f"数据集: {args.dataset}")
    # print(f"特征空间: {args.phis}")

    # 加载数据
    Zs_train = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy")
                for phi in args.phis]
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy")
              for phi in args.phis]
    y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")
    # y_gt_val = np.loadtxt(f"{args.root_dir}/labels/{args.dataset}_val.txt", dtype=float).astype(int)
    # 增强的数据验证和预处理
    Zs_train, Zs_val = robust_data_validation(Zs_train, Zs_val)

    n_tr, C = Zs_train[0].shape[0], datasets_to_c[args.dataset]
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    batch_size = min(args.batch_size, n_tr)

    print(f"训练样本数: {n_tr}, 类别数: {C}")
    # print(f"特征维度: {feature_dims}")
    print(f"批次大小: {batch_size}")

    # 简化的任务编码器
    task_encoder = [nn.Linear(d, C).to(args.device) for d in feature_dims]

    def task_encoding(Zs):
        label_per_space = [safe_softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)]
        labels = torch.mean(torch.stack(label_per_space), dim=0)
        return labels, label_per_space

    # 优化器
    optimizer = torch.optim.AdamW(
        sum([list(task_phi.parameters()) for task_phi in task_encoder], []),
        lr=args.outer_lr, weight_decay=1e-4
    )

    # 简化的学习率调度器
    lr_scheduler = SimpleLRScheduler(optimizer, args.T, args.outer_lr)

    # 内部分类器
    def init_inner():
        W_in = [create_simple_inner_classifier(d, C, args) for d in feature_dims]
        inner_opt = torch.optim.AdamW(
            sum([list(W.parameters()) for W in W_in], []),
            lr=args.inner_lr, weight_decay=1e-4
        )
        return W_in, inner_opt

    W_in, inner_opt = init_inner()

    # Pre-compute neighbors for distillation (与ensemble_en.py中保持一致)
    neighbors_indices = []
    for Z_train in Zs_train:
        # 分别计算图像和文本模态的邻居
        image_features = Z_train[:, :Z_train.shape[1] // 2]
        text_features = Z_train[:, Z_train.shape[1] // 2:]

        image_neighbors = mine_nearest_neighbors(image_features, topk=args.topk)
        text_neighbors = mine_nearest_neighbors(text_features, topk=args.topk)

        neighbors_indices.append({
            'image': image_neighbors,
            'text': text_neighbors
        })

    # 训练状态
    best_cluster_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_model_state = None

    print("开始训练...")
    iters_bar = tqdm(range(args.T))

    for i in iters_bar:
        try:
            # 更新学习率
            current_lr = lr_scheduler.step()

            optimizer.zero_grad()

            # 批次采样
            indices = np.random.choice(n_tr, size=batch_size, replace=n_tr < batch_size)
            Zs_tr = [torch.from_numpy(Z_train[indices]).to(args.device) for Z_train in Zs_train]

            # 任务编码
            labels, label_per_space = task_encoding(Zs_tr)

            # 标签验证
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("无效标签，跳过本次迭代")
                continue

            # 内循环训练，与ensemble_en.py中保持一致
            for idx_inner in range(args.M):
                inner_opt.zero_grad()

                total_inner_loss = torch.tensor(0.0, device=args.device)

                for idx, (w_in, z_tr) in enumerate(zip(W_in, Zs_tr)):
                    # 添加输入验证
                    if torch.isnan(z_tr).any() or torch.isinf(z_tr).any():
                        print(f"输入数据包含NaN或Inf值，跳过")
                        continue

                    # 获取邻居索引
                    batch_neighbors_indices = neighbors_indices[idx]

                    # 随机选择邻居
                    batch_image_neighbors = batch_neighbors_indices['image'][indices]
                    batch_text_neighbors = batch_neighbors_indices['text'][indices]

                    # 随机选择一个邻居
                    neighbor_idx_img = batch_image_neighbors[np.arange(len(indices)),
                    np.random.choice(args.topk, len(indices))]
                    neighbor_idx_text = batch_text_neighbors[np.arange(len(indices)),
                    np.random.choice(args.topk, len(indices))]

                    # 构造邻居特征（完整的特征向量）
                    neighbor_z_tr_img = torch.from_numpy(Zs_train[idx][neighbor_idx_img]).to(args.device)
                    neighbor_z_tr_text = torch.from_numpy(Zs_train[idx][neighbor_idx_text]).to(args.device)

                    # 验证邻居数据
                    if torch.isnan(neighbor_z_tr_img).any() or torch.isinf(neighbor_z_tr_img).any():
                        print(f"邻居图像数据包含NaN或Inf值，跳过")
                        continue
                    if torch.isnan(neighbor_z_tr_text).any() or torch.isinf(neighbor_z_tr_text).any():
                        print(f"邻居文本数据包含NaN或Inf值，跳过")
                        continue

                    # 前向传播并计算蒸馏损失（分别传入图像邻居和文本邻居）
                    # output, distill_loss, _ = w_in(z_tr, neighbor_z_tr_img, neighbor_z_tr_text)
                    output, loss, loss_details = w_in(z_tr, neighbor_z_tr_img, neighbor_z_tr_text)

                    # 如果输出已经是ensemble形式，需要先合并
                    if output.dim() == 3:  # [batch_size, k, output_dim]
                        output = output.mean(dim=1)  # 对ensemble维度求平均

                    # 验证输出和标签
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"模型输出包含NaN或Inf值，跳过")
                        continue

                    # 检查标签范围
                    labels_detached = labels.detach()
                    if torch.any(labels_detached < 0) or torch.any(labels_detached >= C):
                        print(f"标签值超出范围 [0, {C - 1}]，进行裁剪")
                        labels_detached = torch.clamp(labels_detached, 0, C - 1)

                    # 计算分类损失 - 修复：使用软标签时不转换为Long类型
                    classification_loss = F.cross_entropy(output, labels_detached)

                    # 总损失包括分类损失和蒸馏损失
                    total_loss = classification_loss + loss
                    # total_loss = loss
                    # 限制损失值范围
                    total_loss = torch.clamp(total_loss, max=100.0)
                    total_inner_loss += total_loss

                # 检查损失是否正常
                if not torch.isnan(total_inner_loss) and total_inner_loss.requires_grad:
                    # 检查损失值是否合理
                    if total_inner_loss.item() > 1000:  # 设置一个合理的阈值
                        print(f"损失值异常大: {total_inner_loss.item()}")
                    else:
                        total_inner_loss.backward()

                        # 检查梯度
                        has_invalid_grad = False
                        for W in W_in:
                            if has_invalid_gradients(W):
                                has_invalid_grad = True
                                break

                        if not has_invalid_grad:
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(
                                sum([list(W.parameters()) for W in W_in], []), args.grad_clip
                            )
                            inner_opt.step()

            # 更新任务编码器
            optimizer.zero_grad()
            pred_error = torch.tensor(0.0, device=args.device)

            for w_in, z_tr in zip(W_in, Zs_tr):
                output, _, _ = w_in(z_tr, None, None)
                # 如果输出已经是ensemble形式，需要先合并
                if output.dim() == 3:  # [batch_size, k, output_dim]
                    output = output.mean(dim=1).detach()  # 对ensemble维度求平均并detach
                else:
                    output = output.detach()

                # 验证输出
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"任务编码器输出包含NaN或Inf值，跳过")
                    continue

                # 修复：使用软标签时不转换为Long类型
                pred_error += F.cross_entropy(output, labels)

            # 限制pred_error范围
            pred_error = torch.clamp(pred_error, max=100.0)

            # 简化的熵正则化
            entr_reg = torch.tensor(0.0, device=args.device)
            for l in label_per_space:
                probs = l.mean(0)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                entr_reg += -(probs * safe_log(probs)).sum()

            final_loss = pred_error - args.gamma * entr_reg

            if not torch.isnan(final_loss) and final_loss.requires_grad:
                # 检查最终损失
                if final_loss.item() < 1000 and final_loss.item() > -1000:  # 合理的损失值范围
                    final_loss.backward()

                    # 检查任务编码器梯度
                    has_invalid_grad = False
                    for task_phi in task_encoder:
                        if has_invalid_gradients(task_phi):
                            has_invalid_grad = True
                            break

                    if not has_invalid_grad:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            sum([list(task_phi.parameters()) for task_phi in task_encoder], []),
                            args.grad_clip
                        )
                        optimizer.step()

            # 评估
            if (i + 1) % args.eval_freq == 0 or (i + 1) == args.T:
                with torch.no_grad():
                    labels_val, _ = task_encoding(
                        [torch.from_numpy(Z_val).to(args.device) for Z_val in Zs_val]
                    )
                    preds_val = labels_val.argmax(dim=1).cpu().numpy()
                    cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)

                    # 计算NMI和ARI指标
                    nmi = normalized_mutual_info_score(y_gt_val, preds_val)
                    ari = adjusted_rand_score(y_gt_val, preds_val)

                iters_bar.set_description(
                    f"Iter {i + 1}/{args.T}, Loss: {pred_error.item():.3f}, "
                    f"Acc: {cluster_acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, LR: {current_lr:.2e}"
                )

                if cluster_acc > best_cluster_acc:
                    best_cluster_acc = cluster_acc
                    best_nmi = nmi
                    best_ari = ari
                    best_model_state = {
                        f'task_encoder_{j}': task_phi.state_dict()
                        for j, task_phi in enumerate(task_encoder)
                    }
                    print(f"新高准确率: {best_cluster_acc:.4f}, NMI: {best_nmi:.4f}, ARI: {best_ari:.4f}")

        except Exception as e:
            print(f"迭代 {i} 错误: {e}")
            import traceback
            traceback.print_exc()
            # 继续训练而不是停止
            continue

    print(f"\n=== 训练完成 ===")
    print(f"最佳聚类准确率: {best_cluster_acc:.4f}")
    print(f"最佳NMI: {best_nmi:.4f}")
    print(f"最佳ARI: {best_ari:.4f}")

    # 保存模型
    if best_model_state is not None:
        save_path = f"{args.root_dir}/checkpoints/{args.dataset}_best.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"模型已保存: {save_path}")


if __name__ == '__main__':
    run_stable_fixed()
