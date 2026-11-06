# California Housing 回归实验交接

## 运行与复现

项目使用 `data/california_housing.csv`，共 20,640 条样本、8 个原始特征；缺失值与完全重复行均为 0。按 70%/15%/15% 划分训练、验证、测试集，随机种子 42；`data/processed/split_config.json` 保存样本数和行号哈希。运行器会重新生成固定划分，随机森林部分会核对划分哈希与不相交性。

在仓库根目录用当前环境安装 `requirements.txt`，随后运行 `python run_student.py`。它依次执行数据检查和 EDA、Linear/Ridge 与 NumPy Ridge 实验、随机森林基线和五折搜索、最终汇总。解释器及依赖版本记录在 `run_summary.json`，参数和搜索空间见 `results/best_params.json`。

线性模型的插补与标准化位于 Pipeline 内；随机森林的插补与比例特征转换也位于 Pipeline 内，树模型不使用标准化。比例特征分别为 `BedroomRatio`、`RoomsPerPerson`、`HouseholdsApprox`。Ridge 的 alpha 与随机森林参数只在训练集内部使用 5 折交叉验证；候选模型及特征方案按验证集 RMSE 比较。随机森林的选择在测试预测前写入 `results/rf_selection_lock.json`。

## 可供报告使用的结果

| 候选模型 | 特征 | 验证 RMSE | 验证 R² |
|---|---|---:|---:|
| Linear Regression | 原始+比例 | 0.6809 | 0.6442 |
| Ridge | 原始+比例 | 0.6816 | 0.6434 |
| RF 基线 | 原始 | 0.5196 | 0.7927 |
| RF 优化 | 原始 | **0.5176** | **0.7944** |
| RF 基线 | 原始+比例 | 0.5210 | 0.7916 |
| RF 优化 | 原始+比例 | 0.5188 | 0.7934 |

验证集选出的整体模型是原始特征优化随机森林：`n_estimators=80`、`max_depth=20`、`min_samples_leaf=2`、`min_samples_split=2`、`max_features=0.7`。它的五折 CV RMSE 均值约 0.5140、标准差约 0.0113。与 RF 原始特征基线相比，验证 RMSE 仅下降约 0.40%，应如实描述为小幅改善。

锁定模型的测试集 MAE 约 **0.3224**、MSE 约 **0.2385**、RMSE 约 **0.4884**、R² 约 **0.8196**。同一划分下，最佳线性候选的测试 RMSE 约 0.6477。MAE/RMSE 的目标单位为 10 万美元，MSE 为其平方。准确数值以 `deliverables/tables/final_metrics.csv` 为准。

比例特征让线性模型的验证 RMSE 明显下降，但在随机森林上的验证 RMSE 略升。固定其他参数改变树深度时，验证 RMSE 从深度 5 的 0.6791 降至深度 20 的 0.5176；无限深度诊断点为 0.5152。后者只是诊断结果，不据此回改已锁定的方案。随机搜索每个特征方案只抽取 6 组参数，覆盖范围有限。

验证集 permutation importance 最高的是 `MedInc`，其后是 `Latitude`、`Longitude` 和 `AveOccup`。重要性衡量模型预测依赖，不代表因果影响。测试集中目标值 ≥4.99 的 141 条样本 RMSE 约 1.0310，平均预测偏差约 −0.7200；目标上界截断与区域级特征限制了解释。

## 交付文件

- `deliverables/tables/final_metrics.csv`：现有候选的测试指标，`selected_overall_by_validation` 标出整体选择。历史线性实验也保存了各候选的测试指标；模型选择始终依据验证集。
- `deliverables/tables/table_13_all_validation_metrics.csv`：同一验证集上的完整候选比较。
- `deliverables/tables/table_14_all_feature_ablation.csv`：线性模型与随机森林的比例特征消融。
- `results/metrics/ridge_cv_results.csv`、`rf_cv_results.csv`：交叉验证详情。
- `results/predictions/`：带 `row_id`、真实值、预测值和残差的预测文件。
- `deliverables/figures/report/` 与 `figures/ppt/`：分别用于报告与幻灯片的图；完整图注见 `figure_captions.md`。
- `results/models/`：已选模型文件；`deliverables/notes/`：方法、结论、图注、表格摘要及运行摘要。

报告与 PPT 定稿时，需核对数值、数据集合、单位、参数和图注。若只展示一次最终测试评估，应仅展示 `selected_overall_by_validation=True` 的模型行。
