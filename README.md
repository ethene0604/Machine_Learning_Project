# California Housing 回归算法课程项目

本项目在同一份 California Housing 数据上比较 Linear Regression、Ridge Regression
和 Random Forest，并进行五折交叉验证、比例特征消融、模型复杂度与特征重要性分析。
NumPy Ridge 梯度下降实现用于验证算法原理。

## 环境

当前完整实验使用 Python 3.12.7；`requirements.txt` 固定了本次重跑环境中的
NumPy、Pandas、Scikit-learn、Matplotlib、Seaborn 和 Joblib 版本。

```powershell
& "D:\anaconda\python.exe" -m pip install -r requirements.txt
```

在其他机器上，激活对应 Python 环境后使用 `python` 代替上面的解释器路径。

## 一键运行

在仓库根目录执行：

```powershell
& "D:\anaconda\python.exe" run_student.py
```

运行器依次加载 `data/california_housing.csv`、生成固定的 70%/15%/15% 划分，
完成 EDA、Linear/Ridge、NumPy Ridge 和 Random Forest 实验，再汇总指标与图表。
首次生成全套结果约需数分钟。随机种子为 42；模型和特征方案依据训练集内交叉验证
及独立验证集选择，测试集不用于调参。完整配置和版本记录在
`deliverables/notes/run_summary.json`。

如只需基于已有实验文件重建汇总表和全模型比较图，可运行：

```powershell
& "D:\anaconda\python.exe" -m src.reporting
```

此命令不重新拟合模型或生成测试预测。

## 数据与输出

- `data/california_housing.csv`：本地保存的 20,640 条数据，目标 `MedHouseVal` 的单位为 10 万美元。原始来源为 Scikit-learn California Housing 数据集。
- `data/processed/`：固定训练、验证、测试 CSV 与划分行号哈希。
- `results/metrics/`：模型指标、交叉验证、消融、复杂度和重要性结果。
- `results/predictions/`：带 `row_id` 的真实值、预测值及残差。
- `results/best_params.json`、`rf_selection_lock.json`：搜索设置与锁定方案。
- `deliverables/tables/final_metrics.csv`：测试指标汇总；`table_13_all_validation_metrics.csv` 是同一验证集上的候选比较。
- `deliverables/figures/report/` 与 `figures/ppt/`：报告版和 PPT 版图片。
- `deliverables/notes/`：方法、结论、图注、表格摘要、交接说明和统一运行摘要。

## 当前结论

验证集 RMSE 最低的是原始 8 特征的优化随机森林，约 0.5176；锁定后测试集
RMSE 约 0.4884、R² 约 0.8196。比例特征改善了线性模型，但未改善随机森林。
RF 相对基线的调参收益较小，且高价截断区间仍较难预测。详细数值与适用边界见
`deliverables/notes/handoff.md` 和 `result_findings.md`。
