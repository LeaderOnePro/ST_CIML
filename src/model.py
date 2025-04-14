# -*- coding: utf-8 -*-
"""
ST-CIML PyTorch 示例代码 - 模型架构
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# 尝试导入 torch_geometric，如果失败则定义一个占位符
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse # 可能需要，用于转换
    from torch_geometric.data import Data, Batch # 如果使用 DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not found. GNN functionality will be disabled.")
    # 定义一个占位符类，使其在未使用 GNN 时不会引发 NameError
    class GCNConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Warning: GCNConv is a placeholder as torch_geometric is not installed.")
        def forward(self, x, edge_index, *args, **kwargs):
            # 返回输入，使其余部分可以继续（尽管逻辑上不正确）
            print("Warning: Placeholder GCNConv forward pass executed.")
            return x
    TORCH_GEOMETRIC_AVAILABLE = False

# 从 utils 导入设备
from .utils import device

# --- 子模块 --- #

class SpatioTemporalEncoder(nn.Module):
    """
    时空编码器，结合GNN处理空间信息和LSTM处理时间信息。
    """
    def __init__(self, num_nodes: int, dynamic_feat_dim: int, static_feat_dim: int,
                 lulc_feat_dim: int, human_feat_dim: int, initial_feat_dim: int,
                 gnn_hidden_dim: int, lstm_hidden_dim: int, encoder_output_dim: int,
                 use_gnn: bool = True): # 添加 use_gnn 开关
        """
        初始化函数。
        Args:
            ... [其他参数] ...
            use_gnn (bool): 是否使用GNN层。如果为False或torch_geometric不可用，则跳过GNN。
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE # 仅当要求且库可用时才使用GNN

        # 输入特征整合层
        combined_static_dim = static_feat_dim + lulc_feat_dim + initial_feat_dim
        self.input_dim_per_step = dynamic_feat_dim + combined_static_dim + human_feat_dim

        # GNN层 (条件性初始化)
        if self.use_gnn:
            self.gnn1 = GCNConv(self.input_dim_per_step, gnn_hidden_dim)
            self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
            lstm_input_size = gnn_hidden_dim
        else:
            # 如果不使用 GNN，LSTM 的输入直接是组合特征
            lstm_input_size = self.input_dim_per_step
            self.gnn1 = None # 明确设置为 None
            self.gnn2 = None

        # LSTM层
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # 输出层
        self.output_layer = nn.Linear(lstm_hidden_dim, encoder_output_dim)
        self.relu = nn.ReLU()

    def forward(self, dynamic_features: torch.Tensor, static_features: torch.Tensor,
                lulc_map: torch.Tensor, human_activity_feat: torch.Tensor,
                initial_state: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        """
        前向传播。
        Args:
            ... [其他参数] ...
            edge_index (Optional[torch.Tensor]): 形状 [2, E]。如果 use_gnn 为 False，则不需要。
        """
        B, T, N, _ = dynamic_features.shape

        # 1. 准备输入特征
        static_expanded = static_features.unsqueeze(1).expand(B, T, N, -1)
        lulc_expanded = lulc_map.unsqueeze(1).expand(B, T, N, -1)
        initial_expanded = initial_state.unsqueeze(1).expand(B, T, N, -1)
        x_combined = torch.cat([dynamic_features, static_expanded, lulc_expanded,
                                human_activity_feat, initial_expanded], dim=-1)

        # 2. (条件性) 逐时间步应用GNN
        if self.use_gnn:
            if edge_index is None:
                raise ValueError("edge_index must be provided when use_gnn is True.")
            if self.gnn1 is None or self.gnn2 is None: # 安全检查
                 raise RuntimeError("GNN layers were not initialized properly.")

            gnn_output_sequence = []
            # 构建批处理图的边索引 (如果需要)
            # 假设所有样本共享相同的图结构 edge_index [2, E]
            # 为了在批次上使用，需要扩展 edge_index 并调整节点索引
            num_edges = edge_index.shape[1]
            edge_index_list = [(edge_index + i * N) for i in range(B)]
            edge_index_batch = torch.cat(edge_index_list, dim=1).contiguous()

            for t in range(T):
                x_t = x_combined[:, t, :, :].reshape(B * N, -1)
                # GCNConv 通常期望 (num_nodes_in_batch, num_features) 和 (2, num_edges_in_batch)
                h_gnn_t = self.relu(self.gnn1(x_t, edge_index_batch))
                h_gnn_t = self.relu(self.gnn2(h_gnn_t, edge_index_batch))
                h_gnn_t = h_gnn_t.reshape(B, N, -1)
                gnn_output_sequence.append(h_gnn_t)
            processed_sequence = torch.stack(gnn_output_sequence, dim=1) # Shape: B, T, N, gnn_hidden
            lstm_input_features = processed_sequence.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        else:
            # 如果不使用 GNN，直接使用组合特征作为 LSTM 输入
            lstm_input_features = x_combined.permute(0, 2, 1, 3).reshape(B * N, T, self.input_dim_per_step)


        # 3. 应用LSTM处理时间序列
        h_0 = torch.zeros(1, B * N, self.lstm_hidden_dim, device=device)
        c_0 = torch.zeros(1, B * N, self.lstm_hidden_dim, device=device)
        lstm_output, (h_n, c_n) = self.lstm(lstm_input_features, (h_0, c_0))

        # 4. 获取最终编码表示 (使用最后一个时间步的隐藏状态)
        final_encoded_state = h_n.squeeze(0).reshape(B, N, self.lstm_hidden_dim)
        encoder_output = self.relu(self.output_layer(final_encoded_state))

        return encoder_output

class CausalAttributionModule(nn.Module):
    """
    因果归因模块 (简化版)。
    """
    def __init__(self, input_dim: int, num_categories: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_categories)
        )

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        B, N, _ = encoded_state.shape
        reshaped_input = encoded_state.reshape(B * N, -1)
        raw_attributions = self.mlp(reshaped_input)
        raw_attributions = raw_attributions.reshape(B, N, -1)
        return raw_attributions

class InterpretabilityLayer(nn.Module):
    """
    可解释性层，使用自注意力机制。
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        # 确保 embed_dim 可以被 num_heads 整除
        if embed_dim % num_heads != 0:
            # 找到最接近的可整除的 embed_dim (向上取整到 num_heads 的倍数)
            # 或者调整 num_heads。这里我们抛出错误要求用户调整。
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入形状: B, N, embed_dim
        # MultiheadAttention 要求 (batch, seq_len, embed_dim)
        # 这里 N 就是 seq_len
        attn_output, attn_weights = self.attention(node_embeddings, node_embeddings, node_embeddings)
        # attn_output: B, N, embed_dim
        # attn_weights: B, N, N (每个查询节点对所有键节点的注意力)
        return attn_output, attn_weights

class PredictionHead(nn.Module):
    """
    预测头。
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, final_representation: torch.Tensor) -> torch.Tensor:
        B, N, _ = final_representation.shape
        reshaped_input = final_representation.reshape(B * N, -1)
        predictions = self.mlp(reshaped_input)
        predictions = predictions.reshape(B, N, -1) # Shape: B, N, output_dim
        # 如果需要预测多个时间步 (T_pred > 1), output_dim 应该是 F_pred * T_pred
        # 然后在这里 reshape 为 B, T_pred, N, F_pred
        # 例如: predictions = predictions.reshape(B, N, T_pred, F_pred).permute(0, 2, 1, 3)
        return predictions

class AttributionHead(nn.Module):
    """
    归因头。
    """
    def __init__(self, input_dim: int, num_categories: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_categories = num_categories
        if hidden_dim:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_categories)
            )
        else:
            if input_dim != num_categories:
                 print(f"Warning: AttributionHead input_dim ({input_dim}) != num_categories ({num_categories}). Adding a linear layer.")
                 self.mlp = nn.Linear(input_dim, num_categories)
            else:
                 self.mlp = nn.Identity()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, attribution_input: torch.Tensor) -> torch.Tensor:
        B, N, _ = attribution_input.shape
        reshaped_input = attribution_input.reshape(B * N, -1)
        raw_scores = self.mlp(reshaped_input)
        normalized_attributions = self.softmax(raw_scores)
        normalized_attributions = normalized_attributions.reshape(B, N, self.num_categories)
        return normalized_attributions

# --- 主模型 --- #

class ST_CIML(nn.Module):
    """
    完整的 ST-CIML 模型。
    """
    def __init__(self, num_nodes: int, dynamic_feat_dim: int, static_feat_dim: int,
                 lulc_feat_dim: int, human_feat_dim: int, initial_feat_dim: int,
                 gnn_hidden_dim: int, lstm_hidden_dim: int, encoder_output_dim: int,
                 num_attribution_categories: int, causal_hidden_dim: int,
                 attn_heads: int, predict_output_dim: int, predict_hidden_dim: int,
                 attr_head_hidden_dim: Optional[int] = None, use_gnn: bool = True):
        super().__init__()
        self.use_gnn = use_gnn

        # 1. 时空编码器
        self.encoder = SpatioTemporalEncoder(
            num_nodes, dynamic_feat_dim, static_feat_dim, lulc_feat_dim,
            human_feat_dim, initial_feat_dim, gnn_hidden_dim, lstm_hidden_dim,
            encoder_output_dim, use_gnn=self.use_gnn
        )

        # 2. 因果归因模块
        self.causal_module = CausalAttributionModule(
            encoder_output_dim, num_attribution_categories, causal_hidden_dim
        )

        # 3. 可解释性层
        self.interpretability_layer = InterpretabilityLayer(
            encoder_output_dim, attn_heads
        )

        # 4. 预测头
        self.prediction_head = PredictionHead(
            encoder_output_dim, predict_output_dim, predict_hidden_dim
        )

        # 5. 归因头
        self.attribution_head = AttributionHead(
            num_attribution_categories, num_attribution_categories, attr_head_hidden_dim
        )

        # 6. (可选) 物理约束相关层 (临时简单估计器)
        # 这些应该在 __init__ 中定义，而不是在 forward 中反复创建
        self.et_estimator = nn.Linear(encoder_output_dim, 1).to(device)
        self.q_estimator = nn.Linear(encoder_output_dim, 1).to(device)


    def forward(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        模型的前向传播。

        Args:
            data (Dict): 输入数据字典。

        Returns:
            Tuple:
                - predictions (torch.Tensor)
                - normalized_attributions (torch.Tensor)
                - attn_weights (Optional[torch.Tensor])
                - physics_residual (Optional[torch.Tensor])
        """
        # 提取输入
        dyn_feat = data.get('dynamic_features_with_lags') or data['dynamic_features'] # 优先用带lag的
        static_feat = data['static_features']
        lulc = data['lulc_map']
        human_feat = data['human_activity_feat']
        init_state = data['initial_state']
        edge_index = data.get('edge_index', None)

        # 1. 通过编码器
        encoded_output = self.encoder(dyn_feat, static_feat, lulc, human_feat, init_state, edge_index)
        # encoded_output: B, N, encoder_output_dim

        # 2. 通过因果归因模块
        raw_attributions = self.causal_module(encoded_output)
        # raw_attributions: B, N, num_attribution_categories

        # 3. 通过可解释性层
        attn_output, attn_weights = self.interpretability_layer(encoded_output)
        # attn_output: B, N, encoder_output_dim
        # attn_weights: B, N, N

        # 4. 通过预测头 (使用注意力加权后的表示)
        predictions = self.prediction_head(attn_output)
        # predictions: B, N, predict_output_dim

        # 5. 通过归因头
        normalized_attributions = self.attribution_head(raw_attributions)
        # normalized_attributions: B, N, num_attribution_categories

        # 6. 计算物理约束残差 (简化示例)
        physics_residual = None
        try:
            # 检查所需输入是否存在且为 Tensor
            required_keys = ['dynamic_features']
            if not all(k in data and isinstance(data[k], torch.Tensor) for k in required_keys):
                 raise KeyError("Missing required key for physics residual calculation.")

            # 假设原始 dynamic_features 维度: B, T, N, F_dyn
            if data['dynamic_features'].dim() != 4:
                raise ValueError("Unexpected shape for 'dynamic_features' for physics residual.")
            # 假设降水是第0个特征，PET是第1个特征
            precip_last_step = data['dynamic_features'][:, -1, :, 0:1] # Shape: B, N, 1
            pet_last_step = data['dynamic_features'][:, -1, :, 1:2]    # Shape: B, N, 1

            # 使用 attn_output 估计 ET 和 Q
            # 确保 attn_output 形状是 B, N, F
            if attn_output.dim() != 3:
                 raise ValueError("Unexpected shape for attn_output for physics residual.")

            estimated_et = torch.sigmoid(self.et_estimator(attn_output)) * pet_last_step
            estimated_q = torch.relu(self.q_estimator(attn_output))

            # 假设预测目标 (predictions) 是水储量变化 (dS), 形状 B, N, 1
            if predictions.shape[-1] != 1:
                 print("Warning: Physics residual calculation assumes prediction output dim is 1 (dS). Adjust if needed.")
                 predicted_ds = predictions[..., 0:1] # 尝试取第一个特征
            else:
                 predicted_ds = predictions

            physics_residual = predicted_ds - (precip_last_step - estimated_et - estimated_q)
            # physics_residual 形状: B, N, 1

        except (KeyError, IndexError, ValueError, RuntimeError) as e:
            print(f"Could not compute physics residual due to: {e}. Skipping.")
            physics_residual = None

        return predictions, normalized_attributions, attn_weights, physics_residual 