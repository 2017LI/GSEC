# dataset_analysis_generic.py
# 通用数据集分析代码
# 第二部分：对生成的特征进行k-means聚类，
# 然后对生成的每个簇中均匀选出5个样本图片，
# 然后让图片经过llama3.2-vision:11b生成每个图片的描述（使用特定模板形式），
# 然后使用CLIP文本编码器对描述进行编码生成512维特征
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定GPU
# 设置OpenBLAS线程数，解决内存分配问题
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'

import ollama
import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import time
import clip
import argparse

DATASET_CONFIGS = {
    "food101": {"n_clusters": 303, "description": "Food-101 Dataset-101"},
    "dtd": {"n_clusters": 141, "description": "Describable Textures Dataset-47"},
    "stl10": {"n_clusters": 100, "description": "STL-10 Dataset-10"},
    "ucf101": {"n_clusters": 303, "description": "UCF-101 Action Recognition Dataset-101"},
    "imagedogs": {"n_clusters": 120, "description": "ImageNet Dogs Dataset"},
    "aircraft": {"n_clusters": 300, "description": "FGVC Aircraft Dataset-100"},
    "pets": {"n_clusters": 111, "description": "pets Dataset-37"},
    "caltech101": {"n_clusters": 306, "description": "caltech101 Dataset-102"},
    "resisc45": {"n_clusters": 135, "description": "resisc45 Dataset-45"},
    "kitti": {"n_clusters": 50, "description": "kitti Dataset-4"},
    "clevr": {"n_clusters": 24, "description": "clevr Dataset-8"},
    "hatefulmemes": {"n_clusters": 28, "description": "hatefulmemes Dataset-2"},
    "sst": {"n_clusters": 26, "description": "sst Dataset-2"},
    "cars": {"n_clusters": 588, "description": "cars Dataset-196"},
    "sun397": {"n_clusters": 1191, "description": "sun397 Dataset-397"},
    "gtsrb": {"n_clusters": 129, "description": "gtsrb Dataset-43"},
    "eurosat": {"n_clusters": 33, "description": "eurosat Dataset-10"},
    "country211": {"n_clusters": 633, "description": "country211 Dataset-211"},
    "flowers": {"n_clusters": 306, "description": "flower Dataset-102"},
    "cifar10": {"n_clusters": 167, "description": "cifar-10 Dataset-10"},
    "cifar100": {"n_clusters": 300, "description": "cifar-100 Dataset-100"},
    "image10": {"n_clusters": 43, "description": "image10 Dataset-10"},
    "imagenet": {"n_clusters": 3000, "description": "imagenet Dataset-10"}

    }
# 默认数据集名称
DEFAULT_DATASET = "CIFAR10"
# 默认特征空间
DEFAULT_PHIS = "clipvitB32"


def get_dataset_config(dataset_name):
    """获取数据集配置"""
    return DATASET_CONFIGS.get(dataset_name, {"n_clusters": 10, "description": f"{dataset_name} Dataset"})


def load_features(dataset_name, phis=DEFAULT_PHIS):
    """
    加载已提取的特征和图片路径
    """
    # 修改为读取precompute_representations1.py保存的文件路径
    features_file = f'data/representations/{phis}/{dataset_name}_train.npy'
    paths_file = f'data/path/{phis}/{dataset_name}_train.txt'

    if not os.path.exists(features_file) or not os.path.exists(paths_file):
        print(f"错误：未找到特征文件，请先运行precompute_representations1.py")
        print(f"特征文件: {features_file}")
        print(f"路径文件: {paths_file}")
        return None, None

    try:
        features_array = np.load(features_file)
        with open(paths_file, 'r') as f:
            valid_image_paths = [line.strip() for line in f.readlines()]
        print(f"成功加载 {len(valid_image_paths)} 张图片的特征")
        print(f"特征形状: {features_array.shape}")
        return features_array, valid_image_paths
    except Exception as e:
        print(f"加载特征文件时出错: {e}")
        return None, None


def convert_path_format(path):
    """
    将路径转换为统一格式：使用正斜杠并添加./前缀
    """
    # 将反斜杠替换为正斜杠
    normalized_path = path.replace('\\', '/')

    # 如果路径不是以./开头，则添加./前缀
    if not normalized_path.startswith('./'):
        # 移除可能存在的绝对路径前缀
        if normalized_path.startswith('/'):
            normalized_path = '.' + normalized_path
        elif ':' in normalized_path and normalized_path[1] == ':':
            # Windows驱动器路径，如 C:\
            normalized_path = './' + normalized_path.split(':', 1)[1][1:]
        else:
            normalized_path = './' + normalized_path

    return normalized_path


def ensure_dir_exists(file_path):
    """确保文件路径的目录存在"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def perform_clustering(features_array, dataset_name, phis=DEFAULT_PHIS):
    """
    执行K-means聚类
    """
    cluster_labels_file = f'./cluster/{dataset_name}_{phis}_cluster_labels.npy'
    cluster_centers_file = f'./cluster/{dataset_name}_{phis}_cluster_centers.npy'

    # 检查是否已经存在聚类结果
    if os.path.exists(cluster_labels_file) and os.path.exists(cluster_centers_file):
        print("加载已保存的聚类结果...")
        try:
            cluster_labels = np.load(cluster_labels_file)
            cluster_centers = np.load(cluster_centers_file)
            print(f"成功加载聚类结果，共 {len(cluster_centers)} 个簇")
            return cluster_labels, cluster_centers
        except Exception as e:
            print(f"加载聚类结果时出错: {e}")

    # 获取数据集配置
    config = get_dataset_config(dataset_name)
    n_clusters = config["n_clusters"]

    print("执行K-means聚类...")
    # 限制KMeans使用的线程数（新版本sklearn不再支持n_jobs参数）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_array)
    cluster_centers = kmeans.cluster_centers_

    # 确保目录存在后再保存聚类结果
    ensure_dir_exists(cluster_labels_file)
    ensure_dir_exists(cluster_centers_file)
    np.save(cluster_labels_file, cluster_labels)
    np.save(cluster_centers_file, cluster_centers)

    print(f"K-means聚类完成，共 {n_clusters} 个簇")
    return cluster_labels, cluster_centers


def select_representative_samples(cluster_labels, valid_image_paths, features_array, cluster_centers, dataset_name,
                                  phis=DEFAULT_PHIS):
    """
    为每个簇均匀选择5个样本
    """
    representative_samples_file = f'./cluster/{dataset_name}_{phis}_representative_samples.json'

    # 检查是否已经存在代表性样本选择结果
    if os.path.exists(representative_samples_file):
        print("加载已保存的代表性样本...")
        try:
            with open(representative_samples_file, 'r') as f:
                representative_samples_str = json.load(f)
            # 将字符串键转换回整数键，并将路径字符串转换回元组
            representative_samples = {}
            for cluster_id, items in representative_samples_str.items():
                representative_samples[int(cluster_id)] = [(int(idx), path) for idx, path in items]
            print(f"成功加载 {len(representative_samples)} 个簇的代表性样本")
            return representative_samples
        except Exception as e:
            print(f"加载代表性样本时出错: {e}")

    print("为每个簇均匀选择代表性样本...")

    # 将图片按簇分组
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append((i, valid_image_paths[i]))  # 存储索引和路径

    # 为每个簇均匀选择5个样本
    representative_samples = {}
    for cluster_id, items in clusters.items():
        # 获取该簇的中心点
        cluster_center = cluster_centers[cluster_id]

        # 获取该簇中所有样本的特征
        cluster_indices = [item[0] for item in items]
        cluster_features = features_array[cluster_indices]

        # 计算每个样本到簇中心的距离
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)

        # 获取排序后的索引
        sorted_indices = np.argsort(distances)

        # 均匀选择5个样本
        num_samples = len(sorted_indices)
        num_selected = min(5, num_samples)

        if num_samples <= 5:
            # 如果样本数不足或等于5个，选择所有样本
            selected_indices = sorted_indices
        else:
            # 均匀选择5个样本
            # 将排序后的索引分成5个区间，每个区间选择一个样本
            step = num_samples // 5
            selected_indices = []
            for i in range(5):
                start = i * step
                end = start + step if i < 4 else num_samples
                # 选择区间中间的样本
                mid_index = (start + end) // 2
                selected_indices.append(sorted_indices[mid_index])

        # 保存选中的样本（索引和路径）
        representative_samples[int(cluster_id)] = [items[i] for i in selected_indices]

        print(
            f"簇 {cluster_id}: 从 {len(items)} 张图片中均匀选择了 {len(representative_samples[cluster_id])} 张代表性图片")

    # 保存代表性样本选择结果
    # 将元组转换为可序列化的列表，并确保键是Python原生类型
    representative_samples_str = {}
    for cluster_id, items in representative_samples.items():
        # 确保cluster_id是Python原生int类型，而不是numpy类型
        cluster_id_key = int(cluster_id) if isinstance(cluster_id, (np.integer, np.int32, np.int64)) else cluster_id
        representative_samples_str[cluster_id_key] = [(int(idx), path) for idx, path in items]

    # 确保目录存在后再保存文件
    ensure_dir_exists(representative_samples_file)
    with open(representative_samples_file, 'w') as f:
        json.dump(representative_samples_str, f)

    return representative_samples


def generate_image_descriptions(representative_samples, dataset_name, phis=DEFAULT_PHIS):
    """
    为每个簇的代表性图片生成描述，跳过已处理的簇
    """
    image_descriptions_file = f'./cluster/{dataset_name}_{phis}_image_descriptions.npy'

    image_descriptions = {}

    # 检查是否已经存在图片描述结果
    if os.path.exists(image_descriptions_file):
        print("加载已保存的图片描述...")
        try:
            image_descriptions = np.load(image_descriptions_file, allow_pickle=True).item()
            # 确保键是Python原生类型
            image_descriptions = {
                int(k) if isinstance(k, (np.integer, np.int32, np.int64)) else k: v
                for k, v in image_descriptions.items()
            }
            print(f"成功加载 {len(image_descriptions)} 个簇的图片描述")
        except Exception as e:
            print(f"加载图片描述时出错: {e}")
            image_descriptions = {}

    print("\n开始为每个簇的代表性图片生成描述...")

    # 确定需要处理的簇（未生成描述的簇）
    clusters_to_process = []
    for cluster_id in representative_samples.keys():
        # 转换cluster_id为Python原生int类型
        cluster_id = int(cluster_id) if isinstance(cluster_id, (np.integer, np.int32, np.int64)) else cluster_id

        # 如果簇不存在，则需要处理
        if cluster_id not in image_descriptions:
            clusters_to_process.append(cluster_id)
        else:
            print(f"簇 {cluster_id} 已有描述，跳过处理")

    print(f"需要处理 {len(clusters_to_process)} 个簇")

    total_prompt_tokens = 0
    total_output_tokens = 0
    llm_phase_start = time.monotonic() if clusters_to_process else None

    # 为需要处理的簇生成描述
    for cluster_id in clusters_to_process:
        samples = representative_samples[cluster_id]
        print(f"\n处理簇 {cluster_id} (包含 {len(samples)} 个代表性样本)...")

        # 为簇中的代表性样本生成描述（使用特定模板形式）
        cluster_image_descriptions = []
        for i, (idx, path) in enumerate(samples):
            try:
                print(f"  处理图片 {i + 1}/{len(samples)}: {os.path.basename(path)}")

                # 转换路径格式
                converted_path = convert_path_format(path)
                config = get_dataset_config(dataset_name)
                prompt = " Identify and describe the main object in this image. Respond with the format:' This image contains a [object] characterized by [attribute1], [attribute2], and [attribute3]'".

                # 获取图片描述（使用特定模板形式）
                response = ollama.chat(
                    model='llama3.2-vision:11b',
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [converted_path]
                        }
                    ]
                )

                description = response['message']['content']
                pc = getattr(response, "prompt_eval_count", None)
                ec = getattr(response, "eval_count", None)
                if pc is not None:
                    total_prompt_tokens += pc
                if ec is not None:
                    total_output_tokens += ec

                cluster_image_descriptions.append({
                    'index': idx,
                    'path': path,
                    'description': description
                })
                print(f"    描述: {description}")
            except Exception as e:
                print(f"    处理图片 {path} 时出错: {e}")
                continue

        image_descriptions[int(cluster_id)] = cluster_image_descriptions

        # 每处理完一个簇就保存一次，防止中断丢失进度
        # 确保字典键是Python原生类型
        image_descriptions_native = {}
        for k, v in image_descriptions.items():
            image_descriptions_native[int(k) if isinstance(k, (np.integer, np.int32, np.int64)) else k] = v

        # 确保目录存在后再保存文件
        ensure_dir_exists(image_descriptions_file)
        np.save(image_descriptions_file, image_descriptions_native)
        print(f"簇 {cluster_id} 的描述已保存")

    if llm_phase_start is not None:
        llm_phase_seconds = time.monotonic() - llm_phase_start
        total_tokens = total_prompt_tokens + total_output_tokens
        print(
            f"\n[大模型统计] 本阶段 wall 时间: {llm_phase_seconds:.2f} s | "
            f"prompt_eval_count 累计: {total_prompt_tokens} | "
            f"eval_count(输出) 累计: {total_output_tokens} | "
            f"合计约 {total_tokens} tokens（以 Ollama 返回为准；无字段时累计为 0）"
        )

    return image_descriptions


def encode_descriptions_with_clip(image_descriptions, dataset_name, phis=DEFAULT_PHIS):
    """
    使用CLIP文本编码器对描述进行编码生成512维特征
    """
    text_embeddings_file = f'./text/{dataset_name}_{phis}_text_embeddings.npy'

    # 检查是否已经存在编码结果
    if os.path.exists(text_embeddings_file):
        print("加载已保存的文本嵌入...")
        try:
            text_embeddings = np.load(text_embeddings_file)
            print(f"成功加载文本嵌入，形状: {text_embeddings.shape}")
            return text_embeddings
        except Exception as e:
            print(f"加载文本嵌入时出错: {e}")

    print("\n开始使用CLIP对描述进行编码...")

    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50x4", device=device)#ViT-B/32，ViT-L/14、RN50、RN101、RN50x4、
    print(f"CLIP模型已加载，使用设备: {device}")

    # 收集所有有效的描述
    all_valid_descriptions = []
    description_mapping = []  # 记录每个描述来自哪个簇和索引

    for cluster_id in sorted(image_descriptions.keys()):
        descriptions = image_descriptions[cluster_id]
        for i, desc in enumerate(descriptions):
            desc_text = desc['description'].strip()
            # 只处理非空描述，并确保文本长度适中
            if desc_text and len(desc_text) < 600:  # 限制文本长度避免超出模型处理能力
                all_valid_descriptions.append(desc_text)
                description_mapping.append((cluster_id, i))

    print(f"总共收集到 {len(all_valid_descriptions)} 个有效描述")

    # 如果没有有效描述，返回空数组
    if not all_valid_descriptions:
        empty_embeddings = np.zeros((0, 512), dtype=np.float32)
        # 确保目录存在后再保存文件
        ensure_dir_exists(text_embeddings_file)
        np.save(text_embeddings_file, empty_embeddings)
        print(f"文本嵌入已保存，形状: {empty_embeddings.shape}")
        return empty_embeddings

    # 使用CLIP编码所有有效描述
    try:
        text_inputs = clip.tokenize(all_valid_descriptions, truncate=True).to(device)  # 添加truncate=True参数
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            # 转换为numpy数组
            all_embeddings = text_features.cpu().numpy()

        print(f"成功编码 {len(all_valid_descriptions)} 个描述，特征维度: {text_features.shape}")
    except Exception as e:
        print(f"编码描述时出错: {e}")
        # 出错时返回空数组
        empty_embeddings = np.zeros((0, 512), dtype=np.float32)
        # 确保目录存在后再保存文件
        ensure_dir_exists(text_embeddings_file)
        np.save(text_embeddings_file, empty_embeddings)
        print(f"文本嵌入已保存，形状: {empty_embeddings.shape}")
        return empty_embeddings

    # 保存嵌入结果
    # 确保目录存在后再保存文件
    ensure_dir_exists(text_embeddings_file)
    np.save(text_embeddings_file, all_embeddings)
    print(f"文本嵌入已保存，形状: {all_embeddings.shape}")

    return all_embeddings


def save_final_results(cluster_labels, cluster_centers, representative_samples, image_descriptions, text_embeddings,
                       dataset_name, phis=DEFAULT_PHIS):
    """
    保存最终结果
    """
    # 保存聚类结果
    cluster_labels_path = f'./cluster/{dataset_name}_{phis}_cluster_labels.npy'
    cluster_centers_path = f'./cluster/{dataset_name}_{phis}_cluster_centers.npy'

    # 确保目录存在后再保存文件
    ensure_dir_exists(cluster_labels_path)
    ensure_dir_exists(cluster_centers_path)
    np.save(cluster_labels_path, cluster_labels)
    np.save(cluster_centers_path, cluster_centers)

    cluster_detailed_path = f'./cluster/{dataset_name}_{phis}_clusters_detailed.txt'
    # 确保目录存在后再保存文件
    ensure_dir_exists(cluster_detailed_path)
    with open(cluster_detailed_path, 'w', encoding='utf-8') as f:
        for cluster_id, samples in representative_samples.items():
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"Number of representative samples: {len(samples)}\n")
            f.write("Representative samples:\n")
            for idx, path in samples:
                # 转换路径格式用于保存
                converted_path = convert_path_format(path)
                f.write(f"  {converted_path}\n")

            f.write("Image descriptions:\n")
            if cluster_id in image_descriptions:
                for desc in image_descriptions[cluster_id]:
                    f.write(f"  {desc['description']}\n")
            f.write("\n")

    print("\n聚类分析完成!")
    print("结果已保存到:")
    print(f"  - '{cluster_labels_path}'")
    print(f"  - '{cluster_centers_path}'")
    print(f"  - '{cluster_detailed_path}'")
    print(f"  - '{dataset_name}_{phis}_image_descriptions.npy'")
    print(f"  - '{dataset_name}_{phis}_text_embeddings.npy'")


def main(dataset_name=DEFAULT_DATASET, phis=DEFAULT_PHIS):
    """
    主函数
    """
    config = get_dataset_config(dataset_name)
    print(f"处理数据集: {config['description']}")
    print(f"使用特征空间: {phis}")

    # 加载特征
    features_array, valid_image_paths = load_features(dataset_name, phis)

    if features_array is None:
        print("无法加载特征，程序退出")
        return

    # 执行聚类
    cluster_labels, cluster_centers = perform_clustering(features_array, dataset_name, phis)

    # 选择代表性样本
    representative_samples = select_representative_samples(cluster_labels, valid_image_paths, features_array,
                                                           cluster_centers, dataset_name, phis)

    # 生成图片描述
    image_descriptions = generate_image_descriptions(representative_samples, dataset_name, phis)

    # 使用CLIP编码描述生成512维特征
    text_embeddings = encode_descriptions_with_clip(image_descriptions, dataset_name, phis)

    # 保存最终结果
    save_final_results(cluster_labels, cluster_centers, representative_samples, image_descriptions, text_embeddings,
                       dataset_name, phis)

    print("\n第二部分处理完成！所有分析已完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='通用数据集聚类分析工具')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help=f'数据集名称 (默认: {DEFAULT_DATASET})')
    parser.add_argument('--phis', type=str, default=DEFAULT_PHIS,
                        help=f'特征空间 (默认: {DEFAULT_PHIS})')

    args = parser.parse_args()
    main(args.dataset, args.phis)
