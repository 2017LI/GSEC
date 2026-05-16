import torch
import numpy as np
import torch.nn.functional as F
import os
import argparse

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dtd", help="Dataset name")
    parser.add_argument('--phis', type=str, default="clipvitL14", help="Feature space")
    parser.add_argument('--tau', type=float, default=0.005, help="Temperature parameter")
    return parser.parse_args()

# 解析命令行参数
args = parse_args()
dataset = args.dataset
phis = args.phis
tau = args.tau

# 构建文件路径
text_embeddings_file = f'./text/{dataset}_{phis}_text_embeddings.npy'
images_embedding_train_file = f'./data/representations/{phis}/{dataset}_train.npy'
images_embedding_val_file = f'./data/representations/{phis}/{dataset}_val.npy'

# 检查文件是否存在
if not os.path.exists(text_embeddings_file):
    raise FileNotFoundError(f"Text embeddings file not found: {text_embeddings_file}")
if not os.path.exists(images_embedding_train_file):
    raise FileNotFoundError(f"Train images embedding file not found: {images_embedding_train_file}")
if not os.path.exists(images_embedding_val_file):
    raise FileNotFoundError(f"Validation images embedding file not found: {images_embedding_val_file}")

# 读取文本嵌入（代替 nouns_embedding）
text_embeddings = np.load(text_embeddings_file)
text_embeddings = text_embeddings / np.linalg.norm(
    text_embeddings, axis=1, keepdims=True
)
print(f"text_embeddings shape: {text_embeddings.shape}")

# 读取训练集图像嵌入（代替 images_embedding）
images_embedding_train = np.load(images_embedding_train_file)
images_embedding_train = images_embedding_train / np.linalg.norm(
    images_embedding_train, axis=1, keepdims=True
)
print(f"train images_embedding shape: {images_embedding_train.shape}")

# 读取验证集图像嵌入
images_embedding_val = np.load(images_embedding_val_file)
images_embedding_val = images_embedding_val / np.linalg.norm(
    images_embedding_val, axis=1, keepdims=True
)
print(f"validation images_embedding shape: {images_embedding_val.shape}")

# 转换为 PyTorch 张量并移动到 GPU
text_embeddings = torch.from_numpy(text_embeddings).cuda().half()
text_num = text_embeddings.shape[0]
images_embedding_train = torch.from_numpy(images_embedding_train).cuda().half()
train_image_num = images_embedding_train.shape[0]
images_embedding_val = torch.from_numpy(images_embedding_val).cuda().half()
val_image_num = images_embedding_val.shape[0]

def retrieve_embeddings(images_embedding, image_num, prefix):
    """对图像嵌入执行检索过程"""
    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
        similarity = torch.matmul(images_embedding[start:end], text_embeddings.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ text_embeddings).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[{prefix} Completed {min(end, image_num)}/{image_num}]")

    # 合并所有批次的结果
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    return retrieval_embedding

# 处理训练集
print("Processing training set...")
retrieval_embedding_train = retrieve_embeddings(images_embedding_train, train_image_num, "Train")

# 处理验证集
print("Processing validation set...")
retrieval_embedding_val = retrieve_embeddings(images_embedding_val, val_image_num, "Validation")

# 保存结果
output_train_file = f"./CB32/{dataset}_retrieved_nouns_embeddings_train.npy"
output_val_file = f"./CB32/{dataset}_retrieved_nouns_embeddings_val.npy"
os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
np.save(output_train_file, retrieval_embedding_train)
np.save(output_val_file, retrieval_embedding_val)
print(f"Train retrieved text embeddings saved to: {output_train_file}")
print(f"Validation retrieved text embeddings saved to: {output_val_file}")
print(f"Final train retrieval_embedding shape: {retrieval_embedding_train.shape}")
print(f"Final validation retrieval_embedding shape: {retrieval_embedding_val.shape}")

# 新增代码：参照 feature1.py，将图像嵌入与检索到的文本嵌入拼接
print("Concatenating image embeddings with retrieved text embeddings...")

# 确保目录存在
output_dir = f"./data/representations/CB32"
os.makedirs(output_dir, exist_ok=True)

# 将CUDA张量转换回CPU numpy数组用于拼接
images_embedding_train_cpu = images_embedding_train.cpu().numpy()
images_embedding_val_cpu = images_embedding_val.cpu().numpy()

# 拼接训练集特征
combined_features_train = np.concatenate((images_embedding_train_cpu, retrieval_embedding_train), axis=1)
print(f"Combined train features shape: {combined_features_train.shape}")

# 拼接验证集特征
combined_features_val = np.concatenate((images_embedding_val_cpu, retrieval_embedding_val), axis=1)
print(f"Combined validation features shape: {combined_features_val.shape}")

# 保存拼接后的特征
train_output_path = f"{output_dir}/{dataset}_train.npy"
val_output_path = f"{output_dir}/{dataset}_val.npy"

np.save(train_output_path, combined_features_train)
np.save(val_output_path, combined_features_val)

print(f"Combined train features saved to: {train_output_path}")
print(f"Combined validation features saved to: {val_output_path}")
