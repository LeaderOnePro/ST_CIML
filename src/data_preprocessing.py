# -*- coding: utf-8 -*-
"""
ST-CIML PyTorch 示例代码 - 数据模拟与预处理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import copy
from scipy.spatial.distance import cdist # 用于基于距离的连接

# 从 utils 导入设备
from .utils import device

def generate_simulated_data(batch_size: int, time_steps: int, num_nodes: int,
                            num_dynamic_features: int, num_static_features: int,
                            num_lulc_classes: int, num_human_features: int,
                            num_initial_features: int, num_predict_features: int,
                            num_attribution_categories: int,
                            predict_steps: int = 1, use_graph: bool = False) -> Dict:
    """
    生成模拟的ST-CIML输入数据和目标数据。

    Args:
        batch_size (int): 批次大小。
        time_steps (int): 输入时间序列的长度。
        num_nodes (int): 空间节点的数量。
        num_dynamic_features (int): 动态特征的数量。
        num_static_features (int): 静态特征的数量。
        num_lulc_classes (int): LULC类别数量 (用于one-hot)。
        num_human_features (int): 人类活动特征数量。
        num_initial_features (int): 初始状态特征数量。
        num_predict_features (int): 预测目标特征数量。
        num_attribution_categories (int): 归因类别的数量。
        predict_steps (int): 预测未来多少个时间步。
        use_graph (bool): 是否生成图结构数据 (edge_index)。

    Returns:
        Dict: 包含模拟数据的字典。
    """
    simulated_data = {}

    # 动态特征 (例如：气象数据)
    simulated_data['dynamic_features'] = torch.randn(batch_size, time_steps, num_nodes, num_dynamic_features, device=device)

    # 静态特征 (例如：土壤、地形) - 在批次维度上复制，但时间维度上不变
    static_feat = torch.randn(1, num_nodes, num_static_features, device=device).repeat(batch_size, 1, 1)
    simulated_data['static_features'] = static_feat

    # LULC 图 (简化为静态，one-hot编码)
    lulc_indices = torch.randint(0, num_lulc_classes, (batch_size, num_nodes), device=device)
    simulated_data['lulc_map'] = nn.functional.one_hot(lulc_indices, num_classes=num_lulc_classes).float()

    # 人类活动特征 (例如：人口密度，灌溉比例 - 可设为动态或静态，此处模拟为动态)
    simulated_data['human_activity_feat'] = torch.rand(batch_size, time_steps, num_nodes, num_human_features, device=device) * 0.5 # 假设值在0-0.5

    # 初始状态 (例如：初始地下水位)
    simulated_data['initial_state'] = torch.randn(batch_size, num_nodes, num_initial_features, device=device)

    # 目标预测变量 (例如：未来 predict_steps 的水储量变化)
    simulated_data['target_variable'] = torch.randn(batch_size, predict_steps, num_nodes, num_predict_features, device=device)

    # 目标归因 (模拟，例如，假设归因分数和为1)
    raw_attr = torch.rand(batch_size, predict_steps, num_nodes, num_attribution_categories, device=device)
    simulated_data['target_attribution'] = nn.functional.softmax(raw_attr, dim=-1) # 确保和为1

    # 图结构 (可选，模拟一个简单的邻接关系，例如环状连接)
    if use_graph:
        edge_list: List[List[int]] = [] # 指定类型
        for i in range(num_nodes):
            edge_list.append([i, (i + 1) % num_nodes]) # 连接到下一个节点
            edge_list.append([(i + 1) % num_nodes, i]) # 双向连接
        # 为批次中的每个图生成边索引 (假设所有样本共享相同图结构)
        edge_index_single = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        simulated_data['edge_index'] = edge_index_single # Shape: [2, num_edges]

        # 模拟节点坐标用于可视化或距离计算 (如果需要)
        simulated_data['node_coords'] = torch.rand(batch_size, num_nodes, 2, device=device) * 100 # 假设在100x100区域内

    return simulated_data

def standardize_data(data_dict: Dict, fit_scalers: bool = True,
                     scalers: Optional[Dict[str, StandardScaler]] = None) -> Tuple[Dict, Dict[str, StandardScaler]]:
    """
    对输入字典中的数值型张量进行标准化 (Z-score)。

    Args:
        data_dict (Dict): 包含待标准化数据的字典。
        fit_scalers (bool): 是否根据当前数据拟合新的StandardScaler。
                           训练时设为True，验证/测试时设为False。
        scalers (Optional[Dict[str, StandardScaler]]): 预先拟合好的StandardScaler字典。
                                                      验证/测试时提供。

    Returns:
        Tuple[Dict, Dict[str, StandardScaler]]:
            包含标准化后数据的字典 和 使用的StandardScaler字典。
    """
    standardized_data = copy.deepcopy(data_dict) # 避免修改原始数据
    if scalers is None:
        scalers = {}

    # 需要标准化的数值型特征键 (根据实际情况调整)
    keys_to_standardize = ['dynamic_features', 'static_features', 'human_activity_feat', 'initial_state']
    # 目标变量也需要标准化，但通常在损失计算前反标准化预测值，或使用标准化后的目标进行训练
    target_key = 'target_variable'

    for key in keys_to_standardize:
        if key in standardized_data:
            tensor = standardized_data[key]
            original_shape = tensor.shape
            # 将数据展平为 (样本数*节点数*时间步数(如果存在), 特征数) 以便拟合Scaler
            if tensor.dim() == 4: # B, T, N, F
                num_features = original_shape[-1]
                tensor_reshaped = tensor.reshape(-1, num_features).cpu().numpy()
            elif tensor.dim() == 3: # B, N, F
                num_features = original_shape[-1]
                tensor_reshaped = tensor.reshape(-1, num_features).cpu().numpy()
            else:
                print(f"Skipping standardization for {key} due to unexpected dimensions: {tensor.dim()}")
                continue

            if fit_scalers:
                # 检查是否有非零方差
                if tensor_reshaped.shape[0] > 1 and np.std(tensor_reshaped, axis=0).all() > 1e-6 :
                    scaler = StandardScaler()
                    scaler.fit(tensor_reshaped)
                    scalers[key] = scaler
                else:
                    print(f"Warning: Skipping fitting scaler for '{key}' due to zero variance or single sample.")
                    # 创建一个什么都不做的 scaler
                    scaler = StandardScaler(with_mean=False, with_std=False)
                    scalers[key] = scaler # 仍然保存，以便后续转换知道跳过
            elif key not in scalers:
                print(f"Warning: Scaler for '{key}' not provided during inference/validation. Skipping standardization.")
                continue
            else:
                scaler = scalers[key]

            # 应用标准化 (只有在 scaler 实际计算了 mean/std 时才应用)
            if getattr(scaler, 'mean_', None) is not None or getattr(scaler, 'scale_', None) is not None:
                tensor_standardized = scaler.transform(tensor_reshaped)
                # 恢复原始形状
                standardized_data[key] = torch.tensor(tensor_standardized, dtype=torch.float32, device=device).reshape(original_shape)
            else:
                 print(f"Skipping transform for '{key}' as scaler was not fitted or had no variance.")


    # 标准化目标变量 (如果需要)
    if target_key in standardized_data:
        tensor = standardized_data[target_key]
        original_shape = tensor.shape
        if tensor.dim() == 4: # B, T_pred, N, F_pred
            num_features = original_shape[-1]
            tensor_reshaped = tensor.reshape(-1, num_features).cpu().numpy()

            if fit_scalers:
                if tensor_reshaped.shape[0] > 1 and np.std(tensor_reshaped, axis=0).all() > 1e-6 :
                    scaler = StandardScaler()
                    scaler.fit(tensor_reshaped)
                    scalers[target_key] = scaler
                else:
                    print(f"Warning: Skipping fitting scaler for '{target_key}' due to zero variance or single sample.")
                    scaler = StandardScaler(with_mean=False, with_std=False)
                    scalers[target_key] = scaler
            elif target_key not in scalers:
                 print(f"Warning: Scaler for '{target_key}' not provided during inference/validation. Skipping standardization.")
            else:
                scaler = scalers[target_key]
                if getattr(scaler, 'mean_', None) is not None or getattr(scaler, 'scale_', None) is not None:
                    tensor_standardized = scaler.transform(tensor_reshaped)
                    standardized_data[target_key] = torch.tensor(tensor_standardized, dtype=torch.float32, device=device).reshape(original_shape)
                else:
                    print(f"Skipping transform for '{target_key}' as scaler was not fitted or had no variance.")
        elif tensor.dim() == 3: # 可能 B, N, F_pred
            num_features = original_shape[-1]
            tensor_reshaped = tensor.reshape(-1, num_features).cpu().numpy()
            if fit_scalers:
                if tensor_reshaped.shape[0] > 1 and np.std(tensor_reshaped, axis=0).all() > 1e-6 :
                    scaler = StandardScaler()
                    scaler.fit(tensor_reshaped)
                    scalers[target_key] = scaler
                else:
                    print(f"Warning: Skipping fitting scaler for '{target_key}' due to zero variance or single sample.")
                    scaler = StandardScaler(with_mean=False, with_std=False)
                    scalers[target_key] = scaler
            elif target_key not in scalers:
                 print(f"Warning: Scaler for '{target_key}' not provided during inference/validation. Skipping standardization.")
            else:
                scaler = scalers[target_key]
                if getattr(scaler, 'mean_', None) is not None or getattr(scaler, 'scale_', None) is not None:
                    tensor_standardized = scaler.transform(tensor_reshaped)
                    standardized_data[target_key] = torch.tensor(tensor_standardized, dtype=torch.float32, device=device).reshape(original_shape)
                else:
                     print(f"Skipping transform for '{target_key}' as scaler was not fitted or had no variance.")
        else:
             print(f"Skipping standardization for {target_key} due to unexpected dimensions: {tensor.dim()}")


    return standardized_data, scalers


def engineer_features(data_dict: Dict, lag_steps: int = 1) -> Dict:
    """
    执行简单的特征工程，例如创建滞后特征。

    Args:
        data_dict (Dict): 输入数据字典。
        lag_steps (int): 要创建的滞后时间步数。

    Returns:
        Dict: 包含工程化特征的数据字典。
    """
    engineered_data = copy.deepcopy(data_dict)
    if 'dynamic_features' not in engineered_data:
        print("Warning: 'dynamic_features' not found in data_dict for feature engineering.")
        return engineered_data

    dyn_feat = engineered_data['dynamic_features'] # B, T, N, F_dyn
    B, T, N, F_dyn = dyn_feat.shape

    if T <= lag_steps:
        print(f"Warning: Time steps ({T}) <= lag steps ({lag_steps}). Cannot create lagged features.")
        # 返回原始数据，但添加一个空标记以示尝试过
        engineered_data['dynamic_features_with_lags'] = dyn_feat
        return engineered_data

    lagged_features: List[torch.Tensor] = [] # 指定类型
    # 创建动态特征的滞后版本
    for lag in range(1, lag_steps + 1):
        # 将时间步向前移动 'lag' 位，前面用0填充
        lagged = torch.zeros_like(dyn_feat)
        lagged[:, lag:, :, :] = dyn_feat[:, :-lag, :, :]
        lagged_features.append(lagged)

    # 将滞后特征拼接到原始动态特征 (或替换，或作为新特征)
    # 这里作为新特征添加，维度变为 F_dyn * (1 + lag_steps)
    all_dyn_features = [dyn_feat] + lagged_features
    engineered_data['dynamic_features_with_lags'] = torch.cat(all_dyn_features, dim=-1)

    print(f"Created new key 'dynamic_features_with_lags' with shape: {engineered_data['dynamic_features_with_lags'].shape}")

    return engineered_data


def build_graph_adjacency(node_coords: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
    """
    根据节点坐标构建邻接关系 (边索引)。
    这里使用简单的基于距离的阈值方法作为示例。

    Args:
        node_coords (torch.Tensor): 节点坐标张量，形状 [B, N, dims] 或 [N, dims]。
                                    假设所有批次样本共享相同图结构，取第一个样本。
        threshold (float): 定义邻居的最大距离阈值。

    Returns:
        torch.Tensor: 边索引张量，形状 [2, num_edges]，用于 torch_geometric。
    """
    if node_coords.dim() == 3: # B, N, dims
        coords = node_coords[0].cpu().numpy() # 取第一个样本
    elif node_coords.dim() == 2: # N, dims
        coords = node_coords.cpu().numpy()
    else:
        raise ValueError("node_coords must have 2 or 3 dimensions")

    num_nodes = coords.shape[0]
    if num_nodes == 0:
        print("Warning: node_coords contains zero nodes.")
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 计算距离矩阵
    dist_matrix = cdist(coords, coords)

    edge_list: List[List[int]] = [] # 指定类型
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # 避免自环和重复边
            if dist_matrix[i, j] <= threshold:
                edge_list.append([i, j])
                edge_list.append([j, i]) # 添加反向边，因为是无向图

    if not edge_list:
        print("Warning: No edges created with the given threshold. Increase threshold or check coordinates.")
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    print(f"Built graph with {edge_index.shape[1]} edges.")
    return edge_index

# --- 示例用法 (如果直接运行此文件) ---
if __name__ == '__main__':
    # 设置参数
    batch_size = 4
    time_steps = 10
    num_nodes = 50 # 假设50个空间位置/网格单元
    num_dynamic_features = 3 # e.g., Precip, Temp, PET
    num_static_features = 2  # e.g., Soil depth, Elevation
    num_lulc_classes = 5     # e.g., Forest, Urban, Water, Crop, Barren
    num_human_features = 2   # e.g., Population density index, Irrigation ratio
    num_initial_features = 1 # e.g., Initial water storage
    num_predict_features = 1 # e.g., Water storage change
    num_attribution_categories = 3 # e.g., Climate, Human, Natural Variability
    predict_steps = 1
    use_graph_example = True # 设置为 True 以生成 edge_index

    # 生成模拟数据
    example_data = generate_simulated_data(
        batch_size, time_steps, num_nodes, num_dynamic_features,
        num_static_features, num_lulc_classes, num_human_features,
        num_initial_features, num_predict_features, num_attribution_categories,
        predict_steps, use_graph_example
    )

    print("Simulated Data Dictionary Keys:", example_data.keys())
    print("Dynamic Features Shape:", example_data['dynamic_features'].shape)
    if use_graph_example:
        print("Edge Index Shape:", example_data['edge_index'].shape)
        print("Node Coords Shape:", example_data['node_coords'].shape)

    # 标准化数据 (模拟训练阶段)
    standardized_data_train, fitted_scalers = standardize_data(example_data, fit_scalers=True)
    print("Standardized Dynamic Features (mean, std):",
          standardized_data_train['dynamic_features'].mean().item(),
          standardized_data_train['dynamic_features'].std().item())
    print("Fitted Scalers:", fitted_scalers.keys())


    # 特征工程
    engineered_data = engineer_features(standardized_data_train, lag_steps=2)

    # 构建图邻接关系 (如果需要且坐标存在)
    if use_graph_example and 'node_coords' in example_data:
        adjacency_threshold = 30.0 # 示例阈值
        edge_index_built = build_graph_adjacency(example_data['node_coords'], threshold=adjacency_threshold)
        engineered_data['edge_index'] = edge_index_built
        print("Built Edge Index Shape:", engineered_data['edge_index'].shape) 