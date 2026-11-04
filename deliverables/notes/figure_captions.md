# 图片清单与建议图注

所有图片实际结果来自当前项目运行。报告版位于 `deliverables/figures/report/`，
PPT 版位于 `deliverables/figures/ppt/`。

## 01_feature_distributions.png

**建议图注：** 训练集特征与目标变量分布。多个房屋结构变量呈明显右偏和长尾，
目标值在 5.0 附近出现截断。

## 02_feature_boxplots.png

**建议图注：** 训练集特征箱线图。`AveRooms`、`AveBedrms`、`Population` 和
`AveOccup` 存在较多极端值；主实验保留原始样本。

## 03_correlation_heatmap.png

**建议图注：** 训练集 Pearson 相关系数矩阵。`MedInc` 与目标相关最强，
部分房屋结构变量之间存在共线性。

## 04_geographic_price_scatter.png

**建议图注：** 训练样本的经纬度空间分布，颜色表示房屋中位价。房价存在明显
空间聚集，仅使用线性经纬度坐标难以完整表达这种结构。

## 05_model_validation_comparison.png

**建议图注：** 线性模型在验证集上的 RMSE 与 R² 对比。加入比例特征后，
Linear Regression 和 Ridge 的性能均明显提升。

## 06_ridge_alpha_curves.png

**建议图注：** Ridge 正则化参数的五折交叉验证结果。原始特征选择 alpha=3，
比例特征选择 alpha=100；alpha 过大时两种方案均出现欠拟合。

## 07_feature_engineering_ablation.png

**建议图注：** 比例特征前后 RMSE 差值。负值代表误差下降；Linear Regression
和 Ridge 在验证集与测试集上均从比例特征中获益。

## 08_selected_model_coefficients.png

**建议图注：** A 内部最优线性模型的标准化系数。`MedInc` 呈较强正向关联，
纬度、经度为负向关联；这反映模型关联而非因果关系。

## 09_selected_model_split_metrics.png

**建议图注：** 最优线性模型的训练、验证和测试 RMSE、R²。三个集合表现接近，
未出现明显过拟合。

## 10_from_scratch_training_loss.png

**建议图注：** NumPy Ridge 全批量梯度下降的训练目标曲线。学习率 0.05 时
约 3,708 次迭代收敛。

## 11_from_scratch_comparison.png

**建议图注：** NumPy 从零实现与 Scikit-learn Ridge 的 RMSE 对比。两者结果
基本完全一致，验证了自实现梯度与正则系数换算的正确性。

## 12_selected_model_validation_diagnostics.png

**建议图注：** 最优线性模型在验证集上的真实值—预测值和残差图。主体预测接近
y=x，但高价区域存在系统低估。

## 13_selected_model_test_diagnostics.png

**建议图注：** 最优线性模型在测试集上的真实值—预测值和残差图。高价及目标
截断区域的残差明显扩大。

## 14_from_scratch_test_diagnostics.png

**建议图注：** NumPy Ridge 在测试集上的诊断图，其形态与 Scikit-learn Ridge
一致。

## 15_error_by_price_band.png

**建议图注：** 按真实房价区间统计的测试误差和平均偏差。房价高于 3.0 后误差
快速增加，约 5.0 的截断区间被严重低估。
