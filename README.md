# QuickSRNet-Gaming

面向游戏渲染场景的 **QuickSRNet-Small 2×** Fine-Tuning Pipeline。系统复用 Qualcomm 官方网络结构和预训练权重，以严格配对的真实 LR/HR 游戏帧进行全模型微调；不会修改 `qai_hub_models`，也不会在正式训练时联网下载权重。

完整的中文解释、阶段输入输出、张量尺寸及本次实验结果见 [微调管线指南](docs/finetuning_pipeline_guide_zh.md)。

## 环境与预训练权重

建议在项目 Conda 环境中运行：

```powershell
conda activate qsr-gaming
cd F:\CADCG\SRTP\QuickSRNet-Gaming
python .\scripts\00_check_env.py
pip install -r requirements.txt
```

PyTorch 应根据本机 CUDA 单独安装。其余依赖及曲线绘图所需的 Matplotlib 见 `requirements.txt`。

把可信的官方 QuickSRNet-Small 2× checkpoint 放在：

```text
checkpoints/pretrained/quicksrnet_small_2x_checkpoint_float32.pth.tar
```

正式入口沿用已通过 Sanity Check 的本地加载方式：创建官方 Small 2× source model，读取 checkpoint 字典中的 `state_dict`，严格加载 backbone，最后套用 `QuickSRNetSmall` 包装器。因此训练不会调用 `from_pretrained()`，也不会隐式访问网络。

## 数据组织

```text
data/gaming/
├── train/
│   ├── LR/
│   └── HR/
├── val/
│   ├── LR/
│   └── HR/
└── test/
    ├── LR/
    └── HR/
```

- 支持 PNG、JPG、JPEG；目录内可以保留相同的子目录结构。
- LR 与 HR 优先按相对路径和文件 stem 配对，扩展名可以不同。兼容现有 Sanity Check 的 `name_lr.png` / `name_hr.png` 命名。
- 每一对图像都必须是 RGB 可解码图像，且 HR 宽高严格等于 LR 的 2 倍。所有图像在训练启动前检查；缺对、重复 stem、损坏文件或尺寸错误都会给出明确错误。
- 读取后转换为 RGB `[0,1]` Tensor，不使用 ImageNet mean/std，也不转到 `[-1,1]`。
- 正式 LR 按采集结果直接读取；Dataset 不会假设或重新执行 Bicubic 下采样。
- 训练使用同步 Random Crop、水平/垂直翻转和随机 90° 旋转；验证读取完整图像且不做随机增强。为兼容不同分辨率，默认验证 batch size 为 1。

连续游戏帧必须按完整场景或序列划分 Train / Validation / Test，不能把同一序列的相邻帧随机分散到不同 split，否则会造成时序近重复数据泄漏。`data/gaming/test/` 只留给后续最终测试，不参与训练、验证或 best checkpoint 选择。

## 配置

统一配置在 `configs/base.yaml`：

- `model`：模型名、2× scale、官方本地 checkpoint。
- `data`：正式 Train/Validation 路径、LR patch、增强和 DataLoader 设置。
- `training`：epoch、batch size、learning rate、Adam、L1、AMP。
- `validation`：验证 batch、PSNR、best checkpoint 和样例输出。
- `dryrun`：Dry Run 路径及轻量覆盖参数。
- `output`：checkpoint、日志、CSV 指标和可视化样例的根目录。

默认超参数只是可运行初值，尚未针对正式 UE 数据调优。PSNR 以 `[0,1]` RGB、每张图片独立计算后对整个验证集取平均。

## 启动训练

先检查模型加载和原有训练闭环：

```powershell
python .\scripts\01_smoke_test_quicksrnet.py
python .\scripts\02_train_sanity_check.py
```

Dry Run：

```powershell
python .\scripts\03_finetune.py --config configs/base.yaml --dryrun
```

若 `data/dryrun/LR` 和 `HR` 都没有图片，入口会生成一对确定性的合成图。若只有一边有图片，则不会偷偷补数据，而会报告缺失配对。Dry Run 复用训练图片进行验证，仅用于检查数据、反向传播、保存和恢复链路。其 checkpoint 命名为 `best_self_check.pth`，不会被误认为独立验证集上的 `best.pth`。

正式 Fine-Tuning：

```powershell
python .\scripts\03_finetune.py --config configs/base.yaml
```

正式模式只读取配置指定的 `data/gaming/train` 和 `data/gaming/val`。任一目录缺失、为空或存在坏数据都会直接失败，绝不会回退到 Dry Run；test split 不会被读取。

可用 `--device cpu|cuda|auto`、`--epochs N` 和 `--experiment-name NAME` 做少量命令行覆盖。

## 输出与 Resume

每次新运行使用独立实验子目录，避免覆盖旧结果：

```text
checkpoints/finetuned/<run>/
├── last.pth
└── best.pth                 # 正式模式：独立验证集最优
outputs/logs/<run>/train.log
outputs/metrics/<run>/metrics.csv
outputs/samples/<run>/latest_validation_sample.png
```

Dry Run 会以 `best_self_check.pth` 代替 `best.pth`，明确表示该指标来自训练配对自检。

训练 checkpoint 包含 backbone 权重、Optimizer、AMP scaler、已完成 epoch、完整配置、历史指标、当前 best PSNR、运行名和验证集是否独立的标记。`last.pth` 每个 epoch 更新；正式模式按 validation PSNR 保存 `best.pth`。这些文件不会覆盖已有的 `sanity_check.pth`。

从 `last.pth` 继续时，总 epoch 必须大于 checkpoint 中已完成的 epoch：

```powershell
python .\scripts\03_finetune.py --config configs/base.yaml `
  --resume checkpoints/finetuned/<run>/last.pth --epochs 60
```

恢复后会加载模型参数、Optimizer、AMP scaler、epoch、历史指标和 best 指标，并继续写入同一 run 子目录。每次训练结束还会新建模型实例、重新加载 `last.pth`，用验证样本检查推理输出与保存前完全一致（容差 `1e-6`）。

当前训练使用固定学习率，没有启用学习率调度器。Resume 会恢复 checkpoint 中 Adam 的学习率、一阶/二阶矩和更新计数；历史最高 validation PSNR 也会恢复，`best.pth` 只有在分数严格超过历史最高值时才更新。

## 训练曲线

从已保存的逐轮 CSV 导出 Train Loss、Validation PSNR、后期 PSNR 放大图和收敛分析：

```powershell
python .\scripts\04_plot_training_curves.py `
  --metrics outputs/metrics/<run>/metrics.csv `
  --output-dir outputs/analysis/<run>/epoch001-050 --end-epoch 50
```

输出包括组合曲线 PNG/SVG、两张独立曲线 PNG、指定轮数的指标快照、JSON 统计和 `analysis.md`。Train Loss 同时展示原始值与 5 轮移动平均；Validation PSNR 按全分辨率 RGB 图像逐帧计算后取平均。设置 `--end-epoch 100` 可生成续训后的完整曲线和 Epoch 50→100 增量。

## 测试

Dataset 的标准库单元测试不需要额外测试框架：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖正常配对、现有 `_lr`/`_hr` 兼容命名、缺失配对、损坏图像、非严格 2× 尺寸以及同步裁剪后的 Tensor shape。
