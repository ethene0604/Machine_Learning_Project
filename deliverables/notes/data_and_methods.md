# 数据、预处理与模型方法

## 数据集

本实验使用 California Housing 数据集，预测目标为加州 census block group
的房屋中位数价值 `MedHouseVal`，单位为 10 万美元。数据包含 20,640 条样本和
8 个连续数值特征。

数据质量检查结果：

- 缺失值：0；
- 完全重复行：0；
- 目标范围：0.14999～5.00001；
- 目标值接近或等于上界 5.0 的样本存在明显截断；
- `AveRooms`、`AveBedrms`、`Population`、`AveOccup` 和 `MedInc`
  存在长尾或极端值；
- 地理经纬度与房价呈现明显的空间结构。

主实验保留原始极端值，不根据箱线图直接删除样本。缺失值处理使用
`SimpleImputer(strategy="median")`。当前数据没有缺失，因此该步骤在本次运行中是
稳健性保障。

## 数据划分

- 训练集：14,448 条；
- 验证集：3,096 条；
- 测试集：3,096 条；
- 随机种子：42。

模型选择使用验证集 RMSE；Ridge 的正则化参数使用训练集内部 5 折交叉验证选择。
测试集在模型、特征方案和参数确定后，仅用于最终评估。

## 特征工程

在 8 个原始特征之外增加：

```text
BedroomRatio     = AveBedrms / (AveRooms + 1e-9)
RoomsPerPerson   = AveRooms / (AveOccup + 1e-9)
HouseholdsApprox = Population / (AveOccup + 1e-9)
```

这些特征都是不依赖目标值的确定性派生变量。所有模型均使用
`SimpleImputer -> StandardScaler` 的 Pipeline；比例特征在同一条 Pipeline
中生成，确保训练、验证和测试使用完全相同的定义。

## 模型

Linear Regression 使用最小二乘拟合标准化特征与目标之间的线性关系。
Ridge Regression 在平方损失之外加入 L2 正则项：

```text
min ||y - Xw||² + alpha ||w||²
```

搜索范围为 `0.001～10,000` 的对数间隔候选值。最终“原始特征 + 比例特征”
Ridge 选择 `alpha=100`，且 100 不在搜索边界。

## 评价指标

报告 MAE、MSE、RMSE 和 R²。MAE、MSE、RMSE 使用原始目标单位，即
“10 万美元”。例如 RMSE=0.6477 表示约 6.48 万美元。

## 从零实现

NumPy Ridge 使用全批量梯度下降，目标为：

```text
J(w,b) = (1/m) ||Xw + b - y||² + lambda ||w||²
```

截距不参与正则化。为了与 Scikit-learn 的残差平方和目标保持一致，使用
`lambda = alpha / m`。最终选择学习率 0.05，在 3,708 次迭代后收敛。
