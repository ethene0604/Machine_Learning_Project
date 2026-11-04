# California Housing Regression: Student A Experiment

本目录包含同学 A 负责的数据预处理、数据划分、EDA、特征工程、线性回归、
Ridge 回归和 NumPy Ridge 拓展实验。随机森林及最终全模型比较由同学 B 继续完成。

## 环境

本实验结果由以下解释器生成：

```text
C:\ProgramData\anaconda3\python.exe
Python 3.14.6
```

安装依赖：

```powershell
& "C:\ProgramData\anaconda3\python.exe" -m pip install -r requirements.txt
```

## 运行

数据文件已经保存在 `data/california_housing.csv`，可直接运行：

```powershell
Set-Location "D:\Myhomework\机器学习"
& "C:\ProgramData\anaconda3\python.exe" run_student_a.py
```

完整运行约 25 秒，会重新生成所有指标、预测、模型和实验图片。

## Jupyter quickstart

适合开源分享和快速复现的单文件 Notebook：

[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)

打开方式：

```powershell
& "C:\ProgramData\anaconda3\python.exe" -m pip install -r requirements-notebook.txt
& "C:\ProgramData\anaconda3\python.exe" -m jupyter lab notebooks/quickstart.ipynb
```

Notebook 覆盖数据检查、固定划分、特征工程、Linear/Ridge 比较、Ridge 五折调参、
最终预测与指标、误差分析和 NumPy Ridge 从零实现。快速运行结果集中写入
`quickstart_results/`。

## 主要输出

- `data/processed/train.csv`、`validation.csv`、`test.csv`：同学 B 应使用的固定划分；
- `data/processed/split_config.json`：随机种子、样本数和划分索引哈希；
- `results/metrics/`：全部指标、Ridge 交叉验证、特征工程消融和误差分析；
- `results/predictions/`：每个模型在训练集、验证集和测试集上的预测；
- `results/models/`：A 内部最优线性模型和用于拓展的 Ridge 模型；
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

误差指标单位均为 10 万美元。该模型只是 A 负责模型范围内的最优结果；
全体模型最终排名需要等同学 B 完成 Random Forest 后再确定。
