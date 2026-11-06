# Fixed train/validation/test split

这些文件是所有模型共同使用的固定数据划分，不要在建模过程中重新划分。

| Split | Samples | File |
|---|---:|---|
| Train | 14,448 | `train.csv` |
| Validation | 3,096 | `validation.csv` |
| Test | 3,096 | `test.csv` |

划分使用 70%/15%/15% 的嵌套 `train_test_split`，`random_state=42`。
`row_id` 是原始 CSV 中的行号，可用于合并预测和核对样本。

`split_config.json` 额外保存了随机种子、各集合样本数以及索引 SHA256。
随机森林直接读取这些 CSV，并核对行号哈希，以保证各模型使用完全相同的样本。

特征列为：

```text
MedInc, HouseAge, AveRooms, AveBedrms,
Population, AveOccup, Latitude, Longitude
```

目标列为 `MedHouseVal`，单位为 10 万美元。
