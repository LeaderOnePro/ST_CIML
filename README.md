# ST-CIML: 基于PyTorch的ST-CIML架构

## I. 引言

### 目的
本报告旨在提供一个全面的、可运行的Python代码示例，该示例使用PyTorch框架实现了时空因果可解释机器学习（Spatio-Temporal Causal Interpretable Machine Learning, ST-CIML）架构。

### 背景
水文建模和水资源管理领域面临着理解复杂时空动态以及气候变化、人类活动和自然变率之间相互作用的严峻挑战。全球许多地区的水资源短缺问题日益严重，这不仅受到气候变化（如降水模式改变、蒸发加剧、冰雪融化）的影响，也受到直接人类活动（如地下水过度开采、农业灌溉、大坝建设、人口增长、城市化、污染）的显著驱动。此外，厄尔尼诺等自然变率也扮演着重要角色。区分物理性缺水（资源总量不足）和经济性缺水（基础设施或管理能力不足）对于制定有效的应对策略至关重要。因此，迫切需要开发不仅能准确预测水文状态（如储水量、径流），还能提供因果归因和可解释性的模型，以揭示变化的驱动因素。机器学习，特别是深度学习方法，在水文预测方面展现出巨大潜力，但其"黑箱"特性和缺乏物理一致性限制了其在科学发现和决策支持中的应用。

### ST-CIML 架构概述 (保留核心组件说明)

ST-CIML架构旨在应对这些挑战，其核心组件包括：

*   **时空编码器 (SpatioTemporalEncoder):** 学习输入的动态和静态特征的时空联合表示。
*   **因果归因模块 (CausalAttributionModule):** (简化版) 尝试将预测结果归因于不同的驱动因素类别（如气候、人类活动）。
*   **可解释性层 (InterpretabilityLayer):** 提供模型内部关注机制的洞察，例如通过注意力权重。
*   **预测头 (PredictionHead) 和 归因头 (AttributionHead):** 分别输出最终的水文状态预测值和归一化的归因分数。

通过结合这些模块，并整合物理约束（示例），ST-CIML力求在预测性能、因果洞察和模型透明度之间取得平衡。

## 免责声明

**重要:** 本报告提供的代码使用 **模拟数据** 进行演示，旨在作为一个基础模板。实际应用需要根据具体研究问题和数据进行调整，包括：

*   使用真实世界数据 (ERA5, ESA CCI, GLDAS, GRACE, 人口/用水数据等)。
*   进行彻底的数据预处理（清洗、插值、标准化、特征工程）。
*   选择和优化模型架构（例如，不同的GNN层、RNN变体、注意力机制）。
*   进行严格的模型验证（时空交叉验证）和超参数调整（学习率、网络维度、损失权重等）。
*   使用合适的评估指标 (如 KGE, RMSE)。

**特别需要指出的是:**

1.  **简化归因:** 本示例中的"因果归因模块"是对真实因果推断的 **简化表示**。其输出应被理解为模型学习到的、与预测相关的 **关联性**，而非严格意义上的因果关系证明。在复杂的时空系统中进行严格的因果推断本身就充满挑战。
2.  **模型局限性:** 机器学习模型本身存在局限性，如泛化能力、对极端事件的预测能力以及物理过程的解释能力等。
3.  **解释性警示:** 用户应谨慎解释本示例中简化模块（CausalAttributionModule, InterpretabilityLayer）的输出。注意力权重和归因分数反映的是模型在学习预测任务时发现的内部关联模式，它们可以提供有价值的假设，但不能直接等同于经过验证的因果关系或物理机制。必须结合领域知识、敏感性分析和其他解释方法进行验证。

## 结论

本仓库提供了一个基于PyTorch的ST-CIML架构的详细Python代码示例，涵盖了从数据模拟、预处理、模型构建（包括时空编码器、简化因果归因模块、注意力可解释层、预测头和归因头）、物理信息整合、组合损失函数定义到基本训练循环的完整流程。该架构旨在整合机器学习的预测能力与环境科学研究所需的物理一致性和可解释性。

## 使用说明与适配

**模板性质:** 必须强调，所提供的代码是一个 **基础模板**，使用模拟数据进行演示。用户在应用于实际研究时，必须进行以下适配：

1.  **环境设置:**
    *   安装 Python (推荐使用 Conda 管理环境)。
    *   安装 PyTorch (`conda install pytorch torchvision torchaudio cudatoolkit=xx.x -c pytorch` 或 `pip install torch torchvision torchaudio`)。
    *   (如果使用GNN) 安装 PyTorch Geometric (`pip install torch_geometric` 或根据官方文档)。
    *   安装其他依赖: `pip install numpy scikit-learn scipy`。
    *   (推荐) 创建 `environment.yml` 或 `requirements.txt` 文件来管理依赖。

2.  **数据替换与预处理 (`src/data_preprocessing.py`, `data/`):**
    *   将模拟数据 (`generate_simulated_data`) 替换为真实的、经过仔细预处理和标准化的环境数据集。
    *   将真实数据存储在 `data/raw` 目录下，预处理后的数据（如Tensor）存储在 `data/processed`。
    *   确保输入数据加载后符合模型期望的字典格式（键名和张量维度，见 `generate_simulated_data` 函数文档）。
    *   调整 `standardize_data` 函数中的 `keys_to_standardize` 以匹配您的特征。实现健壮的 Scaler 保存/加载机制（例如，使用 `joblib` 或 `pickle` 将训练集上拟合的 `fitted_scalers` 保存到 `output/scalers/`）。
    *   根据需要增强 `engineer_features` 函数，添加领域相关的特征（如计算SPEI、滞后项、移动平均等）。
    *   如果使用GNN，确保 `build_graph_adjacency` 函数（或类似函数）能根据您的空间数据（坐标、邻接矩阵等）正确构建 `edge_index`。

3.  **配置 (`config/`):**
    *   (推荐) 将模型超参数（如隐藏层维度、注意力头数）、训练参数（学习率、批次大小、轮数、损失权重 `lambda_attr`, `lambda_physics`）和数据相关配置（文件路径、特征名称）移至YAML配置文件 (`config/*.yaml`)。
    *   修改 `src/train.py` 或创建一个新的主脚本来加载这些配置。

4.  **模型定制 (`src/model.py`):**
    *   根据具体的数据特性（如网格 vs. 不规则站点）、空间依赖性（如是否需要GNN）和研究目标，选择或修改编码器 (`SpatioTemporalEncoder`)、归因模块 (`CausalAttributionModule`) 和可解释性层 (`InterpretabilityLayer`) 中的具体网络结构（例如，尝试不同的GNN层如 GAT，或使用 Transformer 代替 LSTM，或采用更复杂的注意力机制）。
    *   调整 `ST_CIML` 类中的物理约束计算部分，实现更符合研究区域和过程的、更精确的物理约束方程。确保输入数据包含计算所需的所有变量。

5.  **训练与评估 (`src/train.py`, `src/evaluate.py`):**
    *   **数据加载:** 实现 PyTorch `Dataset` 和 `DataLoader` 类 (`src/datasets.py`) 来高效地加载和批处理您的真实数据，而不是依赖 `generate_simulated_data`。
    *   **验证与测试:** 实现独立的验证和测试流程。在 `train_model` 函数中加入验证步骤（例如，在每轮结束后在验证集上评估性能）。创建一个 `src/evaluate.py` 脚本，用于加载训练好的模型（来自 `output/trained_models/`）并在测试集上进行最终评估。
    *   **交叉验证:** 设计合理的时空交叉验证方案。
    *   **超参数调优:** 使用工具（如 Optuna, Ray Tune）或手动进行全面的超参数搜索。
    *   **评估指标:** 使用适合水文应用的评估指标（如 KGE, NSE, RMSE）。
    *   **早停:** 实现早停（Early Stopping）机制以防止过拟合（监控验证集损失）。
    *   **模型保存:** 在训练过程中定期保存模型检查点，并保存验证集上性能最佳的模型。

6.  **因果与解释性深化:**
    *   如果需要更严格的因果分析或更深入的模型解释，应考虑结合更专业的因果推断库（如 DoWhy, CausalML）或外部XAI工具（如 SHAP, LIME）对训练好的模型进行后处理分析。将分析结果保存在 `output/attributions/` 或 `output/plots/`。

7.  **运行代码:**
    *   主要的训练入口点是 `src/train.py`。可以直接运行 `python -m src.train` (如果将 `src` 目录添加到 `PYTHONPATH` 或在 `ST_CIML` 根目录下运行)。
    *   (推荐) 创建顶层脚本（如 `run_training.py`）来处理配置加载、模型实例化和调用 `train_model`。

**运行示例 (使用模拟数据):**

```bash
python -m src.train
```

(请确保您的 Python 环境已设置，并且您位于 `ST_CIML` 项目的根目录下)

这个命令将使用 `src/train.py` 中定义的默认参数和模拟数据生成器运行一个简短的训练演示。 