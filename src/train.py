# -*- coding: utf-8 -*-
"""
ST-CIML PyTorch 示例代码 - 训练脚本
"""

import torch
import torch.optim as optim
import torch.nn as nn # 导入 nn 以便在物理约束中使用临时估计器
from typing import Dict, Callable

# 从本项目模块导入
from .utils import device, set_seed, SEED
from .data_preprocessing import (generate_simulated_data,
                                 standardize_data, engineer_features,
                                 build_graph_adjacency)
from .model import ST_CIML
from .loss_functions import calculate_combined_loss

def train_model(model: nn.Module, train_data_generator: Callable, optimizer: optim.Optimizer,
                num_epochs: int, steps_per_epoch: int, batch_size: int,
                lambda_attr: float, lambda_physics: float, attr_loss_type: str,
                data_gen_args: dict, use_graph: bool, lag_steps: int,
                build_graph_args: Optional[Dict] = None,
                preprocessing_args: Optional[Dict] = None):
    """
    一个基本的模型训练循环。

    Args:
        model (nn.Module): 待训练的ST-CIML模型。
        train_data_generator (callable): 用于生成训练批次数据的函数。
        optimizer (optim.Optimizer): 优化器。
        num_epochs (int): 训练轮数。
        steps_per_epoch (int): 每轮训练的步数（批次数）。
        batch_size (int): 批次大小。
        lambda_attr (float): 归因损失权重。
        lambda_physics (float): 物理损失权重。
        attr_loss_type (str): 归因损失类型。
        data_gen_args (dict): 传递给数据生成函数的参数字典。
        use_graph (bool): 是否在数据生成或预处理中启用图相关功能。
        lag_steps (int): 特征工程中使用的滞后步数。
        build_graph_args (Optional[Dict]): 传递给 build_graph_adjacency 的参数 (如 threshold)。
        preprocessing_args (Optional[Dict]): 其他预处理参数 (暂未使用，但可扩展)。
    """
    model.train() # 设置模型为训练模式

    # 假设标准化器在外部拟合和传递，或者我们在这里为模拟数据拟合一次
    # 为简化，每次训练运行时都基于第一批数据拟合scaler，这不理想，仅用于演示
    print("Fitting scalers on the first batch for demonstration purposes...")
    first_batch = train_data_generator(batch_size=batch_size, **data_gen_args)
    _, fitted_scalers = standardize_data(first_batch, fit_scalers=True)
    print("Scalers fitted.")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            # 1. 生成/加载一批训练数据
            batch_data = train_data_generator(batch_size=batch_size, **data_gen_args)

            # 2. 数据预处理
            #   a. 标准化 (使用已拟合的 scaler)
            batch_data_std, _ = standardize_data(batch_data, fit_scalers=False, scalers=fitted_scalers)
            #   b. 特征工程 (创建滞后项)
            batch_data_eng = engineer_features(batch_data_std, lag_steps=lag_steps)
            #   c. 图构建 (如果需要且未在生成器中完成)
            if use_graph and 'node_coords' in batch_data_eng and 'edge_index' not in batch_data_eng:
                if build_graph_args is None:
                    build_graph_args = {'threshold': 20.0} # 默认阈值
                try:
                    batch_data_eng['edge_index'] = build_graph_adjacency(batch_data_eng['node_coords'], **build_graph_args)
                except Exception as e:
                    print(f"Error building graph adjacency at epoch {epoch+1}, step {step+1}: {e}. Skipping step.")
                    continue

            # 确保所有必需的输入都在字典中，并移到正确的设备 (生成器应该已经做了)
            required_model_inputs = ['static_features', 'lulc_map', 'human_activity_feat', 'initial_state']
            if 'dynamic_features_with_lags' in batch_data_eng:
                required_model_inputs.append('dynamic_features_with_lags')
            else:
                 required_model_inputs.append('dynamic_features')
            if model.use_gnn:
                 required_model_inputs.append('edge_index')

            input_data_for_model = {}
            all_keys_present = True
            for k in required_model_inputs:
                 if k not in batch_data_eng:
                      print(f"Error: Missing required input key '{k}' for model at epoch {epoch+1}, step {step+1}. Skipping.")
                      all_keys_present = False
                      break
                 if isinstance(batch_data_eng[k], torch.Tensor):
                     input_data_for_model[k] = batch_data_eng[k].to(device)
                 else:
                      # 可能是 None 或其他类型，模型forward需要处理
                      input_data_for_model[k] = batch_data_eng[k]
            if not all_keys_present: continue

            # 还要确保目标值存在
            if 'target_variable' not in batch_data_eng:
                 print(f"Error: Missing 'target_variable' at epoch {epoch+1}, step {step+1}. Skipping.")
                 continue
            targets = batch_data_eng['target_variable'].to(device)
            target_attributions = batch_data_eng.get('target_attribution')
            if target_attributions is not None:
                 target_attributions = target_attributions.to(device)
            # 为了计算物理损失，原始动态特征也需要 (如果使用了lagged特征)
            if 'dynamic_features' in batch_data: # 从原始数据获取
                 input_data_for_model['dynamic_features'] = batch_data['dynamic_features'].to(device)
            elif 'dynamic_features' in input_data_for_model: # 如果没有lagged，它就在这里
                 pass
            else:
                  print(f"Warning: Missing original 'dynamic_features' for physics loss calculation at epoch {epoch+1}, step {step+1}.")


            # 3. 清零梯度
            optimizer.zero_grad()

            # 4. 模型前向传播
            try:
                predictions, attributions, attn_weights, physics_residual = model(input_data_for_model)
            except Exception as e:
                print(f"Error during forward pass at epoch {epoch+1}, step {step+1}: {e}")
                # 打印输入形状以帮助调试
                for k, v in input_data_for_model.items():
                    if isinstance(v, torch.Tensor):
                        print(f"Shape of {k}: {v.shape}")
                    else:
                        print(f"Input {k}: {v}")
                print(f"Targets shape: {targets.shape}")
                if target_attributions is not None:
                     print(f"Target Attributions shape: {target_attributions.shape}")
                continue # 跳过此批次

            # 5. 计算损失
            try:
                loss = calculate_combined_loss(
                    predictions, targets, attributions, target_attributions,
                    physics_residual, lambda_attr, lambda_physics, attr_loss_type
                )
            except ValueError as e:
                 print(f"Error calculating loss at epoch {epoch+1}, step {step+1}: {e}")
                 print(f"Preds: {predictions.shape}, Targs: {targets.shape}, Attrs: {attributions.shape}, PhysRes: {physics_residual.shape if physics_residual is not None else None}")
                 continue # 跳过此批次
            except Exception as e:
                 print(f"Unexpected error calculating loss at epoch {epoch+1}, step {step+1}: {e}")
                 continue # 跳过此批次

            # 6. 反向传播
            loss.backward()

            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 7. 更新参数
            optimizer.step()

            epoch_loss += loss.item()

            # 打印中间步骤损失 (可选)
            if (step + 1) % 10 == 0:
                 print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{steps_per_epoch}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

# --- 主执行块 --- #
if __name__ == '__main__':
    # --- 设置参数 (从模拟数据部分复制并调整) ---
    set_seed(SEED) # 确保种子被设置

    batch_size = 4
    time_steps = 10
    num_nodes = 50
    num_dynamic_features = 3
    num_static_features = 2
    num_lulc_classes = 5
    num_human_features = 2
    num_initial_features = 1
    num_predict_features = 1
    num_attribution_categories = 3
    predict_steps = 1
    use_graph_config = True # 控制是否启用GNN相关功能
    lag_steps_config = 2      # 特征工程滞后步数
    adjacency_threshold_config = 30.0 # 图构建阈值

    # --- 模型参数 --- #
    # 注意 dynamic_feat_dim 需要考虑滞后项
    dynamic_dim_input_to_model = num_dynamic_features * (1 + lag_steps_config)

    model_params = {
        'num_nodes': num_nodes,
        'dynamic_feat_dim': dynamic_dim_input_to_model,
        'static_feat_dim': num_static_features,
        'lulc_feat_dim': num_lulc_classes,
        'human_feat_dim': num_human_features,
        'initial_feat_dim': num_initial_features,
        'gnn_hidden_dim': 64,
        'lstm_hidden_dim': 128,
        'encoder_output_dim': 64,
        'num_attribution_categories': num_attribution_categories,
        'causal_hidden_dim': 32,
        'attn_heads': 4, # 确保 encoder_output_dim (64) 能被 attn_heads (4) 整除
        'predict_output_dim': num_predict_features * predict_steps, # 如果多步预测，需要调整
        'predict_hidden_dim': 64,
        'attr_head_hidden_dim': None, # 直接用Softmax
        'use_gnn': use_graph_config
    }

    # --- 训练参数 --- #
    learning_rate = 1e-3
    num_epochs_config = 5 # 减少示例轮数以便快速运行
    steps_per_epoch_config = 20 # 减少示例步数
    lambda_attr_train = 0.2
    lambda_physics_train = 0.1
    attr_loss_type_train = 'constraint'

    # --- 数据生成器参数 --- #
    data_gen_args_train = {
        'time_steps': time_steps,
        'num_nodes': num_nodes,
        'num_dynamic_features': num_dynamic_features,
        'num_static_features': num_static_features,
        'num_lulc_classes': num_lulc_classes,
        'num_human_features': num_human_features,
        'num_initial_features': num_initial_features,
        'num_predict_features': num_predict_features,
        'num_attribution_categories': num_attribution_categories,
        'predict_steps': predict_steps,
        'use_graph': use_graph_config # 传递给生成器，以便生成 node_coords
    }

    # --- 图构建参数 (如果使用) --- #
    build_graph_args_train = None
    if use_graph_config:
        build_graph_args_train = {'threshold': adjacency_threshold_config}

    # --- 实例化模型和优化器 --- #
    st_ciml_model = ST_CIML(**model_params).to(device)
    print("\nST-CIML Model Structure:")
    # print(st_ciml_model) # 打印模型结构可能很长

    optimizer = optim.Adam(st_ciml_model.parameters(), lr=learning_rate)

    # --- 运行训练循环 --- #
    print("\nStarting training loop...")
    train_model(
        st_ciml_model,
        generate_simulated_data, # 使用模拟数据生成器
        optimizer,
        num_epochs_config,
        steps_per_epoch_config,
        batch_size,
        lambda_attr_train,
        lambda_physics_train,
        attr_loss_type_train,
        data_gen_args_train,
        use_graph=use_graph_config,
        lag_steps=lag_steps_config,
        build_graph_args=build_graph_args_train
    )

    print("\nDemo training finished.") 