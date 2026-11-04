# 同学 A 交接清单

## 已完成

- [x] 官方数据本地准备与完整性检查；
- [x] 20,640 条样本、8 个特征、缺失值和重复行检查；
- [x] 70%/15%/15% 固定划分，并保存 `row_id` 与索引哈希；
- [x] EDA 图片与数据概况表；
- [x] 中位数插补、标准化和无泄漏 Pipeline；
- [x] 三个比例特征及原始/工程特征消融实验；
- [x] Linear Regression 与 Ridge Regression；
- [x] Ridge 五折交叉验证和 alpha 曲线；
- [x] 训练、验证、测试指标与预测文件；
- [x] NumPy Ridge 梯度下降从零实现；
- [x] 真实值—预测值、残差、系数和高价截断误差分析；
- [x] 报告版图片、PPT 版图片、表格、图注和结果解读。

## 交给同学 B

同学 B 直接使用以下固定文件，不要重新划分：

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
data/processed/split_config.json
```

可在代码中复用：

```python
from src.features import AddRatioFeatures, build_preprocessor
```

RF 验证和测试结果建议使用 `rf_*` 文件名，不要覆盖 `linear_*` 和 `ridge_*`。
RF 的参数选择和特征工程消融必须继续只使用训练集与验证集；A 的测试结果不能
用于反向修改 RF 参数。

## 交给报告/PPT 同学

- 报告图片：`deliverables/figures/report/`；
- PPT 图片：`deliverables/figures/ppt/`；
- 完整数据表：`deliverables/tables/`；
- 方法说明：`deliverables/notes/data_and_methods.md`；
- 可直接使用的结论：`deliverables/notes/result_findings.md`；
- 图注：`deliverables/notes/figure_captions.md`；
- 格式化指标表：`deliverables/notes/table_summary.md`。

## 需要共同确认

1. A 的“最优模型”只代表当前线性模型范围，RF 最优结果出来后才能确定全体最优；
2. 所有误差指标单位均为 10 万美元；
3. 不把验证集指标写成测试集指标；
4. 不将标准化系数解释为因果效应；
5. 报告和 PPT 完成后，由两位实验同学核对数字、图注、单位、模型参数和结论。
