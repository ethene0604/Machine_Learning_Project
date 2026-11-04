# California Housing Regression: Student A and B Experiments

本目录包含同学 A 的数据、线性模型与 NumPy Ridge 实验，以及同学 B 的
随机森林基线、五折调参、特征工程消融、复杂度和特征重要性分析。

## 环境

同学 A 原始结果记录的解释器：

```text
C:\ProgramData\anaconda3\python.exe
Python 3.14.6
```

该路径在当前机器已不可用。同学 B 结果由 `D:\anaconda\python.exe`
（Python 3.12.7、scikit-learn 1.5.1）生成，具体版本见
`deliverables/notes/student_b_run_summary.json`。B 的运行依赖单独记录在
`requirements-b.txt`，A 的环境见 `requirements.txt`。安装 B 的依赖：

```powershell
& "D:\anaconda\python.exe" -m pip install -r requirements-b.txt
```

## 运行

数据文件已经保存在 `data/california_housing.csv`，可直接运行：

```powershell
Set-Location "D:\My_projects\Machine_Learning_Project"
& "D:\anaconda\python.exe" run_student_b.py
& "D:\anaconda\python.exe" build_student_b_handoff.py
```

`run_student_b.py` 使用现有固定划分，不会重新划分或覆盖 A 的指标。
它会重新运行 RF 搜索和锁定模型的测试集评估；若只需重建汇总表和比较图，
仅运行 `build_student_b_handoff.py`，不会重新预测测试集。

## Jupyter quickstart

适合开源分享和快速复现的单文件 Notebook：

[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)

打开方式：

```powershell
& "D:\anaconda\python.exe" -m pip install -r requirements-notebook.txt
& "D:\anaconda\python.exe" -m jupyter lab notebooks/quickstart.ipynb
```

Notebook 覆盖数据检查、固定划分、特征工程、Linear/Ridge 比较、Ridge 五折调参、
最终预测与指标、误差分析和 NumPy Ridge 从零实现。快速运行结果集中写入
`quickstart_results/`。

## 主要输出

- `data/processed/train.csv`、`validation.csv`、`test.csv`：同学 B 应使用的固定划分；
- `data/processed/split_config.json`：随机种子、样本数和划分索引哈希；
- `results/metrics/`：全部指标、Ridge 交叉验证、特征工程消融和误差分析；
- `results/predictions/`：RF 所有候选的训练/验证预测，锁定模型的测试预测，以及 A 的预测；
- `results/models/`：A 的线性模型、Ridge 模型和 B 锁定的 RF 模型；
- `deliverables/figures/report/`：300 dpi 报告图片；
- `deliverables/figures/ppt/`：适合 PPT 使用的高分辨率图片；
- `deliverables/tables/`：可直接放入报告的 CSV 表格；
- `deliverables/notes/`：数据方法说明、图注、结果解读和交接清单。

## 当前结果

A 负责的候选中，验证集 RMSE 最低的是“原始特征 + 三个比例特征”的
Linear Regression。其在独立测试集上的结果为：

| MAE | MSE | RMSE | R² |
|---:|---:|---:|---:|
| 0.4735 | 0.4195 | 0.6477 | 0.6828 |

MAE 和 RMSE 的单位为 10 万美元，MSE 为其平方。该模型只是 A 负责模型范围内的最优结果；
全体模型的验证集最低 RMSE 是优化随机森林（原始特征），0.5176；
锁定后测试集 RMSE 为 0.4884、R² 为 0.8196。汇总见
`deliverables/tables/final_metrics.csv`，B 的交接说明见
`deliverables/notes/student_b_handoff.md`。
