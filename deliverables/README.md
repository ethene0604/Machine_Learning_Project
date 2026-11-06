# 报告与 PPT 交付包

本目录汇集完整实验生成的表格、图表和说明。所有指标均来自 `run_student.py`
的实际运行结果，正式报告中请保留数据集合名称和目标单位。

- `tables/final_metrics.csv`：测试集指标，`selected_overall_by_validation` 标记整体选中模型。
- `tables/table_13_all_validation_metrics.csv`：同一验证集上的所有候选。
- `tables/table_14_all_feature_ablation.csv`：线性模型与随机森林的比例特征消融。
- `figures/report/`：报告版 300 dpi PNG；`figures/ppt/`：幻灯片版 220 dpi PNG。
- `notes/data_and_methods.md`：数据、预处理、模型和评价方法。
- `notes/result_findings.md`：实验结论、解释与局限。
- `notes/figure_captions.md`：01–21 图的建议图注。
- `notes/table_summary.md`：可直接参考的指标摘要。
- `notes/handoff.md`：实验设置、主要结果、产物清单与核对事项。
- `notes/run_summary.json`：统一运行的版本、划分、模型选择和时长记录。

报告与 PPT 定稿前，请以 CSV 的完整精度核对手写数字和图注。
