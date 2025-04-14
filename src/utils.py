# -*- coding: utf-8 -*-
"""
ST-CIML PyTorch 示例代码 - 工具函数
"""

import torch
import numpy as np
import os
import random

# 设置随机种子以保证可复现性
def set_seed(seed: int = 42):
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        # 确保卷积操作的可复现性，但可能牺牲性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# 设备设置
def get_device() -> torch.device:
    """获取可用的计算设备 (GPU 或 CPU)。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# --- 初始化 ---
SEED = 42
set_seed(SEED)
device = get_device()

print(f"PyTorch version: {torch.__version__}")
# print(f"Torch Geometric version: {torch_geometric.__version__}") # 如果使用GNN 