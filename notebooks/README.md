# Quickstart notebook

打开：

```powershell
Set-Location "D:\Myhomework\机器学习"
& "C:\ProgramData\anaconda3\python.exe" -m jupyter lab notebooks/quickstart.ipynb
```

不打开界面、直接执行并保存输出：

```powershell
& "C:\ProgramData\anaconda3\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  notebooks/quickstart.ipynb
```

Notebook 会使用仓库中的本地数据；若数据不存在，会尝试从 scikit-learn
官方地址下载。执行结果写入 `quickstart_results/`，不会覆盖主实验的
`results/` 或 `deliverables/`。
