"""Fixed-protocol Synthetic LR diagnostic on Scene_1_02; never trains or reads test data."""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw
import scipy
from scipy.ndimage import gaussian_filter
import skimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PairedImageDataset
from src.metrics import SSIM_PARAMETERS, rgb_psnr_per_image, rgb_ssim
from src.trainer import build_quicksrnet_small_from_local_checkpoint, resolve_device
from src.utils import create_logger, seed_everything

COMMIT = "d09d2b0404d10f71a7640a87e9d5e5257b028802"
OFFICIAL_URL = f"https://github.com/quic/aimet-model-zoo/blob/{COMMIT}/aimet_zoo_torch/quicksrnet/"
DEFAULT_CAPTURED = ROOT / "outputs/verification/val-scene1-02-20261019-140251"
DEFAULT_OFFICIAL = Path.home() / ".qaihm/qai-hub-models/models/aimet_zoo/v1/quic_aimet-model-zoo_git/aimet_zoo_torch/quicksrnet"
METHODS = ("bicubic", "original", "finetuned")
CONDITIONS = ("captured", "synthetic")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the prior evaluation's serialization, hashing and fixed, score-independent sample plan.
previous = load_module("existing_validation", ROOT / "scripts/05_verify_validation.py")
write_json, fingerprint, snapshot = previous.write_json, previous.fingerprint, previous.snapshot
hwc, image_from_float, font, ROIS = previous.hwc, previous.image_from_float, previous.font, previous.ROIS


def synthetic_u8(hr_u8, resize):
    """Official public evaluation recipe, including its final truncation (not rounding)."""
    assert hr_u8.dtype == np.uint8 and hr_u8.shape == (1080, 1920, 3)
    resized = resize(hr_u8.astype(np.float64), scalar_scale=0.5, method="bicubic", mode="vec")
    return np.uint8(np.clip(resized, 0.0, 255.0))


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(lr, hr, models, pool):
    predictions = {"bicubic": F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=False)}
    predictions.update({name: model(lr) for name, model in models.items()})
    arrays, metrics = {}, {}
    target = hwc(hr)
    for name, output in predictions.items():
        if output.shape != hr.shape or not torch.isfinite(output).all():
            raise ValueError(f"Invalid {name} prediction")
        output = output.clamp(0, 1)
        metrics[name + "_psnr_db"] = rgb_psnr_per_image(output, hr).item()
        arrays[name] = hwc(output)
    futures = {name: pool.submit(rgb_ssim, array, target) for name, array in arrays.items()}
    for name, future in futures.items():
        metrics[name + "_ssim"] = future.result()
    assert all(math.isfinite(value) for value in metrics.values())
    return metrics, arrays


def strip(images, labels, title, width, crop=None, rois=()):
    """Pixel-preserving native full panels or identical 2x nearest ROI enlargement."""
    gap, header = 16, 136 if crop is None else 108
    panels = []
    for image in images:
        panel = image.crop(crop) if crop else image.copy()
        if crop is None:
            draw = ImageDraw.Draw(panel)
            for index, (_, box) in enumerate(rois):
                draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), outline=previous.ROI_COLORS[index], width=4)
        if panel.width != width:
            panel = panel.resize((width, round(panel.height * width / panel.width)), Image.Resampling.NEAREST)
        panels.append(panel)
    canvas = Image.new("RGB", (len(panels)*width+(len(panels)+1)*gap, header+panels[0].height+gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, fill="#172033", font=font(32 if crop is None else 21, True))
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = gap + i*(width+gap)
        for j, line in enumerate(label.split("\n")):
            draw.text((x, 54+j*30 if crop is None else 43+j*26), line, fill="#172033", font=font(28 if crop is None else 20, j == 0))
        canvas.paste(panel, (x, header))
    return canvas


def alignment_probe(captured, synthetic, rois):
    """Diagnostic only: do not register, shift or crop evaluation inputs using its result."""
    cap = gaussian_filter(captured.mean(axis=2), sigma=1.5)
    syn = gaussian_filter(synthetic.mean(axis=2), sigma=1.5)
    cap = np.stack(np.gradient(cap), axis=-1)
    syn = np.stack(np.gradient(syn), axis=-1)
    regions = [("interior", (32, 32, 928, 508))]
    regions += [(name, tuple(int(v//2) for v in box)) for name, box in rois]
    checks = []
    for name, (x0,y0,x1,y1) in regions:
        a = syn[y0:y1,x0:x1].ravel().astype(np.float64)
        a -= a.mean()
        scores = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                # Positive offset samples a later captured pixel at the same reference position.
                b = cap[y0+dy:y1+dy,x0+dx:x1+dx].ravel().astype(np.float64)
                b -= b.mean()
                ncc = float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15))
                scores.append({"captured_sampling_offset_yx": [dy,dx], "gradient_ncc": ncc})
        checks.append({"region": name, "lr_box_xyxy": [x0,y0,x1,y1], "zero_offset_ncc": scores[12]["gradient_ncc"],
                       "best": max(scores, key=lambda item: item["gradient_ncc"]), "all_offsets": scores})
    return checks


def save_samples(out, frame, captured, synthetic, hr, cap_arrays, syn_arrays, cap_row, syn_row):
    stem = f"Scene_1_02_frame_{frame:04d}"
    hr_image = image_from_float(hr)
    nearest = lambda a: image_from_float(a).resize(hr_image.size, Image.Resampling.NEAREST)
    groups = {
        "A_inputs": ([nearest(captured), nearest(synthetic), hr_image],
                     ["Captured LR\nNearest 2x; display only", "Synthetic LR\nNearest 2x; display only", "HR Ground Truth\nUnchanged RGB target"]),
    }
    for condition, arrays, row in (("captured",cap_arrays,cap_row),("synthetic",syn_arrays,syn_row)):
        labels = [f"{label}\n{row[m+'_psnr_db']:.3f} dB | SSIM {row[m+'_ssim']:.4f}" for m,label in zip(METHODS,("Bicubic 2x","Original QuickSRNet","Fine-tuned (Epoch 91)"))]
        groups["B_"+condition] = ([image_from_float(arrays[m]) for m in METHODS]+[hr_image], labels+["HR Ground Truth\nUnchanged RGB target"])
    files, layouts, previews = [], [], {}
    for group, (images, labels) in groups.items():
        for roi_name, box in [("full", None)]+[(f"roi_{chr(65+i)}",box) for i,(_,box) in enumerate(ROIS[frame])]:
            width = 1920 if box is None else 512
            detail = "Native 1080p; boxed ROIs" if box is None else f"HR ROI {box}; nearest 2x zoom; scores are FULL FRAME"
            canvas = strip(images,labels,f"{stem} | {group} | {detail}",width,box,ROIS[frame])
            path = out / "samples" / f"{stem}_{group}_{roi_name}.png"
            canvas.save(path, compress_level=3)
            files.append(str(path.relative_to(out)))
            layouts.append({"file": files[-1], "frame":frame, "group":group,"crop_hr_xyxy":box,"columns":len(images),"panel_width":width,"gap":16,"panel_y":136 if box is None else 108,"panel_height":1080 if box is None else 512})
            if (group == "A_inputs" and box is None) or (group != "A_inputs" and roi_name == "roi_A"):
                previews[group] = canvas.resize((1600,round(canvas.height*1600/canvas.width)),Image.Resampling.LANCZOS)
    return files, layouts, previews


def official_equivalence_check(official_dir, resize):
    # Execute ONLY the official pure pairing function, avoiding its optional plotting imports.
    tree = ast.parse((official_dir/"dataloader/utils.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_hr_lr_pair")
    scope = {"np":np,"torch":torch,"imresize":resize}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(official_dir/"dataloader/utils.py"),"exec"),scope)
    rng = np.random.default_rng(42)
    image = rng.integers(0,256,(48,64,3),dtype=np.uint8)
    official_lr, official_hr = scope["create_hr_lr_pair"](image,2)
    ours = np.uint8(np.clip(resize(image.astype(np.float64),scalar_scale=0.5,method="bicubic",mode="vec"),0,255))
    ours_tensor = torch.from_numpy(ours.transpose(2,0,1).copy()).float().div(255)
    assert torch.equal(official_lr,ours_tensor)
    assert torch.equal(official_hr,torch.from_numpy(image.transpose(2,0,1).copy()).float().div(255))
    # Unequal constant channels detect accidental channel mixing; high-frequency case exercises AA.
    constant = np.empty((48,64,3),np.uint8); constant[:] = [32,128,224]
    constant_lr, _ = scope["create_hr_lr_pair"](constant,2)
    assert torch.max(torch.abs(constant_lr-torch.tensor([32,128,224]).view(3,1,1)/255)) <= 1/255 + 1e-7
    return scope["create_hr_lr_pair"], {"random_rgb_pair_bitwise_equal":True,"rgb_channel_order_checked":True,"crop_on_real_hr":False}


def write_readme(out, summary):
    table = ["| LR 来源 | 方法 | PSNR (dB) | SSIM | PSNR 相对 Bicubic |", "|---|---|---:|---:|---:|"]
    for row in summary["comparison"]:
        table.append(f"| {row['condition']} | {row['method']} | {row['mean_psnr_db']:.6f} | {row['mean_ssim']:.6f} | {row['delta_psnr_vs_bicubic_db']:+.6f} |")
    facts = summary["diagnostic_deltas"]
    text = f"""# QuickSRNet-Small 2× Synthetic LR 诊断实验

仅使用 Scene_1_02 的 240 张固定 HR；Captured 条件复用先前完整评估，Synthetic 条件重新计算全部 240 帧。微调权重为 100 Epoch 实验的全程最佳 Epoch 91。未访问 Scene_1_05，未训练、调参或进行速度 Benchmark。

## 结果（实验事实）

{chr(10).join(table)}

{chr(10).join('- '+line for line in summary['observations'])}

差值单位：PSNR 为 dB、SSIM 为绝对差。dB 差不是画质百分比。完整跨条件差值、逐帧胜负计数见 summary.json；两条件各 240 行见 per_image_metrics.csv。

## 官方约定核对与预先固定的方案

- [官方发布说明]({OFFICIAL_URL}QuickSRNet.md)明确发布模型在 DIV2K 上训练，float32 权重为其验证准确率最佳模型。
- [官方公开评估预处理]({OFFICIAL_URL}dataloader/utils.py)使用 RGB uint8 HR → float64 [0,255] → [MATLAB 风格 imresize]({OFFICIAL_URL}dataloader/imresize.py) (scale=0.5, method='bicubic', mode='vec') → clip [0,255] → np.uint8 **截断** → float32 /255。沿用原代码的截断，即使注释称模拟 imwrite，也不把它改为四舍五入。
- 下采样核为 cubic a=-0.5；基础支撑宽度 4，0.5 下采样时扩展至 8，内置抗混叠（没有独立 antialias 开关）；按半像素中心映射、对称重复边界、逐维归一化权重，float64 中间计算。真实 HR 尺寸能整除 2，不发生裁剪。
- 本次直接加载该固定 commit 的本地 imresize.py，并快照原文件及哈希。将包装逻辑与官方 create_hr_lr_pair 在随机 RGB 样例及真实首帧上做逐元素等价核对。
- **未核实该 checkpoint 训练时完整的 LR 退化配方。** 上述证据确认的是官方公开**评估预处理**，不能据此宣称已复现训练退化。论文 §3.3 确认 RGB [0,1]，§5 的游戏效果来自在游戏数据上另行训练，不能当成当前 DIV2K checkpoint 的游戏结果。[论文](https://arxiv.org/html/2303.04336)。
- 下采样方案和 0000/0060/0120/0180/0239 的样例及原有 ROI 坐标在评分前固定；只运行这一种生成方法，没有根据结果选择核。

## 数据、模型和评分

数据经既有 PairedImageDataset 按相对 stem 配对。读取全部 480 张原图核对尺寸、PNG 模式、alpha、色彩标签；frame_0000 至 frame_0239 无重复或遗漏。Pillow convert('RGB') 后按保存的编码 RGB 值评分；不额外线性化、不应用 ICC 色彩变换。元数据不能证明采集时物理色彩管线完全相同，详见 data_checks.json。

使用现有模型构造器和严格 state_dict 加载，其初始化方式与已安装官方 QAI 实现相同；项目官方权重与历史评分时的 SHA256 完全相同。QAI 缓存副本的额外核验状态见 summary.json（若系统禁止读取，不改变权限，也不宣称核对成功）。微调 checkpoint epoch=91、配置总轮数=100。模型均 eval()、inference_mode()、float32，无 AMP、TTA 或分块。模型结构、checkpoint 路径与 SHA256 见 summary.json。

复用 src.metrics：RGB 三通道整图 PSNR，peak=1，无边缘裁切，先逐帧计算 dB 再取算术平均；输出 clamp [0,1]。SSIM 使用 skimage.metrics.structural_similarity，参数 {SSIM_PARAMETERS}，通道等权后逐帧等权；Gaussian reflect、truncate=3.5，标量平均有效窗口中心（11×11 窗导致 SSIM map 边界排除 5 像素），输入图本身不裁剪。所有指标在导出展示 PNG 前计算。

两条件的 Bicubic **上采样**均为 torch.nn.functional.interpolate(size=(1080,1920),mode='bicubic',align_corners=False,antialias=False)，然后 clamp。这是既有基线；它的核与本次官方风格**下采样**实现不完全相同，仍有 Bicubic 家族匹配偏差。不要把下采样抗混叠设置套用到旧上采样基线上。

Captured 结果从 {summary['captured_provenance']['directory']} 的 per_image_metrics.csv 复制，先验证全部原图、权重、共享代码哈希和指标参数，再验证 CSV 均值与旧 summary 一致。仅对 5 个固定展示帧重新推理并核对旧逐帧指标，容差 PSNR 0.0001 dB / SSIM 0.000001；不重新跑 Captured 全量评估。

Synthetic LR 缓存为 synthetic_lr/Scene_1_02/frame_XXXX.png，RGB uint8 无损 PNG；评分重新解码缓存后转换 float32 /255，确保缓存等于实际输入。所有 240 张缓存哈希见 synthetic_manifest.json。

## 输入差异和对齐诊断

lr_difference_metrics.csv 包含同编号 Captured 与 Synthetic 的 LR 像素 PSNR/SSIM，以及两者灰度均方梯度能量。它们只是输入差异指标，**不是超分质量指标**；未经独立渲染标定，不能把差值全部归因为某个核、AA 或噪声。

alignment_checks.json 在预先固定的 5 帧、每帧内区与两个既有 ROI 上，以 sigma=1.5 平滑的平均 RGB 灰度梯度，在 ±2 LR 像素网格内计算 NCC。记录零偏移及所有 25 个偏移；不据此改变任何评分图像。该探针能发现明显整数位移，不能证明不存在亚像素偏移、时域抖动、局部非刚性差异、LOD/曝光差异或时间不同步。正偏移表示在 Captured 中取更后的位置与同一 Synthetic 位置比较。

## 合理推测与尚未验证的结论

{summary['interpretation']}

当前只对单条连续验证序列做诊断；240 帧不是 240 个独立场景，且微调模型已用本序列选模。没有自然图像对照组、其他场景或独立测试，因此不能分解 Image Domain Gap 与 Degradation Gap 的因果贡献比例，也不能宣布真实游戏泛化性能。Synthetic 从 HR 派生，改变信息保留方式与任务难度；与同家族插值匹配可能偏向 Bicubic。真实游戏表现仍应以 Captured 条件及后续独立测试为准。

## 可视化

samples/ 共 5 组固定帧。A_inputs 是 Captured LR / Synthetic LR（均仅 nearest 2× 展示）/ HR；B_captured 和 B_synthetic 是 Bicubic / Original / Fine-tuned / HR。每组保留原生全图和两个相同 HR 坐标的 256×256 ROI，以 nearest 同样放大 2×；标题中的指标是整帧指标。sample_plan.json 与 figure_layouts.json 记录全部坐标和版式。

- [输入全图概览](samples/overview_inputs.png)
- [重建 ROI 概览：每帧先 Captured 后 Synthetic](samples/overview_reconstruction_rois.png)
- [首帧输入 ROI A](samples/Scene_1_02_frame_0000_A_inputs_roi_A.png)
- [首帧 Captured 重建 ROI A](samples/Scene_1_02_frame_0000_B_captured_roi_A.png)
- [首帧 Synthetic 重建 ROI A](samples/Scene_1_02_frame_0000_B_synthetic_roi_A.png)

## 复现命令（PowerShell）

在项目根目录执行，输出目录必须全新，已有结果不会覆盖：

```powershell
& 'D:/anaconda3/envs/qsr-gaming/python.exe' scripts/06_diagnose_synthetic_lr.py --output-dir outputs/diagnostics/synthetic_lr/reproduction-001 --captured-dir outputs/verification/val-scene1-02-20261019-140251 --device cuda
```

默认从已安装 QAI 缓存定位固定官方源码；其他机器可用 --official-dir 指向该 commit 的 aimet_zoo_torch/quicksrnet 目录。软件版本和实际命令见 summary.json。source/ 保留本次脚本、共享代码及官方预处理快照。

protected_files_before.json / after.json 记录原始验证数据、已有权重、日志、训练曲线和历史验证结果的哈希、大小、mtime；本次前后全部保持一致。独立质量审计见 quality_checks.json。
"""
    (out/"README.md").write_text(text,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--captured-dir",type=Path,default=DEFAULT_CAPTURED)
    parser.add_argument("--official-dir",type=Path,default=DEFAULT_OFFICIAL)
    parser.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda")
    args = parser.parse_args()
    out, captured_dir, official_dir = (ROOT/args.output_dir).resolve(), (ROOT/args.captured_dir).resolve(), args.official_dir.resolve()
    out.relative_to(ROOT/"outputs/diagnostics/synthetic_lr")
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    # Reject a historical directory or source path that would escape the declared validation scope.
    captured_dir.relative_to(ROOT/"outputs/verification")
    captured_summary = json.loads((captured_dir/"summary.json").read_text(encoding="utf-8"))
    assert captured_summary["status"] == "PASS" and captured_summary["dataset"]["pairs"] == 240
    assert captured_summary["dataset"]["sequence"] == "Scene_1_02"
    assert captured_summary["evaluation"]["ssim_parameters"] == SSIM_PARAMETERS
    assert captured_summary["evaluation"]["bicubic"] == {"library":"torch.nn.functional.interpolate","size_hw":[1080,1920],"mode":"bicubic","align_corners":False,"antialias":False,"float_interpolation":True,"clamp":[0,1]}
    for name in ("src/metrics.py","src/dataset.py","src/trainer.py","scripts/05_verify_validation.py"):
        assert fingerprint(ROOT/name)["sha256"] == captured_summary["source_sha256"][str(Path(name))]
    with (captured_dir/"per_image_metrics.csv").open(encoding="utf-8-sig",newline="") as handle:
        captured_csv = list(csv.DictReader(handle))
    assert len(captured_csv) == 240 and len({row['name'] for row in captured_csv}) == 240
    old_means = {row['method']:row for row in captured_summary['comparison']}
    for method in METHODS:
        for metric, mean_key in (("psnr_db","mean_psnr_db"),("ssim","mean_ssim")):
            assert abs(sum(float(row[method+'_'+metric]) for row in captured_csv)/240-old_means[method][mean_key]) < 1e-12
    lr_root, hr_root = ROOT/"data/gaming/val/LR", ROOT/"data/gaming/val/HR"
    for path in (lr_root,hr_root):
        assert sorted(p.name for p in path.iterdir() if p.is_dir()) == ["Scene_1_02"]
    dataset = PairedImageDataset(lr_root,hr_root,scale_factor=2,lr_patch_size=None,augment=False)
    expected = [f"scene_1_02/frame_{i:04d}" for i in range(240)]
    assert [p.key for p in dataset.pairs] == [r['name'] for r in captured_csv] == expected
    assert all(p.lr_size == (960,540) and p.hr_size == (1920,1080) for p in dataset.pairs)
    old_manifest = json.loads((captured_dir/"protected_files_before.json").read_text(encoding="utf-8"))
    # Only the explicit old manifest and known report/analysis files are read; never scan test/train images.
    protected = [Path(p) for p in old_manifest]
    protected += [p for p in captured_dir.rglob('*') if p.is_file()]
    protected += [p for p in (ROOT/"outputs/analysis"/previous.RUN).rglob('*') if p.is_file()]
    assert not any("scene_1_05" in str(p).lower() or "data\\gaming\\test" in str(p).lower() for p in protected)
    before = snapshot(protected)
    assert all(before[name] == value for name,value in old_manifest.items()), "Captured inputs or checkpoints changed since prior scoring"
    for pair, row in zip(dataset.pairs,captured_csv):
        assert Path(row['lr_path']) == pair.lr_path and Path(row['hr_path']) == pair.hr_path
        assert str(pair.lr_path) in old_manifest and str(pair.hr_path) in old_manifest
    official_path = Path(captured_summary['checkpoints']['official']['path'])
    fine_path = Path(captured_summary['checkpoints']['finetuned']['path'])
    for key, path in (("official",official_path),("finetuned",fine_path)):
        assert before[str(path)]['sha256'] == captured_summary['checkpoints'][key]['sha256']
    fine = torch.load(fine_path,map_location="cpu",weights_only=True)
    assert fine['epoch'] == 91 and fine['config']['training']['epochs'] == 100 and fine['run_name'] == previous.RUN and fine['scale_factor'] == 2
    cache_checkpoint = Path.home()/".qaihm/qai-hub-models/models/quicksrnetsmall/v2"/official_path.name
    try:
        cache_crosscheck = {"path":str(cache_checkpoint),"readable":True,"sha256":fingerprint(cache_checkpoint)['sha256']}
        assert cache_crosscheck['sha256'] == fingerprint(official_path)['sha256']
    except PermissionError:
        cache_crosscheck = {"path":str(cache_checkpoint),"readable":False,"reason":"Operating system denied access; project checkpoint hash matches historical experiment; no permissions changed"}
    out.mkdir(parents=True)
    (out/"samples").mkdir(); (out/"source").mkdir()
    synthetic_root = out/"synthetic_lr/Scene_1_02"; synthetic_root.mkdir(parents=True)
    write_json(out/"protected_files_before.json",before)
    sources = [Path(__file__),ROOT/"scripts/05_verify_validation.py",ROOT/"src/metrics.py",ROOT/"src/dataset.py",ROOT/"src/trainer.py"]
    sources += [official_dir/"dataloader/imresize.py",official_dir/"dataloader/utils.py",official_dir/"QuickSRNet.md"]
    installed = Path(sys.executable).parent/"Lib/site-packages/qai_hub_models"
    sources += [installed/"models/quicksrnetsmall/model.py",installed/"models/templates/super_resolution/model.py",installed/"utils/aimet/repo.py"]
    source_info = {}
    for i,path in enumerate(sources):
        destination = out/"source"/f"{i:02d}_{path.name}"
        shutil.copyfile(path,destination)
        source_info[str(path)] = {**fingerprint(path),"snapshot":str(destination.relative_to(out))}
    # Freeze the exact external implementation and run that immutable snapshot.
    resize = load_module("official_diagnostic_imresize",out/"source/05_imresize.py").imresize
    official_pair, equivalence = official_equivalence_check(official_dir,resize)
    seed_everything(42); torch.set_num_threads(4)
    logger = create_logger("synthetic_diagnostic."+out.name,out/"evaluation.log")
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    plan = {"fixed_before_scoring":True,"frames":list(ROIS),"rois":ROIS,"coordinates":"HR xyxy zero-based half-open","downsampling":"Official public evaluation imresize bicubic AA + clipped uint8 truncation; single preselected recipe", "alignment_probe":"fixed 5 frames; 3 regions; sigma=1.5 gray gradient NCC; offsets -2..2; no correction"}
    write_json(out/"sample_plan.json",plan)
    device = resolve_device(args.device)
    models = {}
    for name in ("original","finetuned"):
        model = build_quicksrnet_small_from_local_checkpoint(official_path,scale_factor=2)
        if name == "finetuned": model.model.load_state_dict(fine['state_dict'],strict=True)
        models[name] = model.float().to(device).requires_grad_(False).eval()
    assert sum(p.numel() for p in models['original'].parameters()) == 22860
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,pin_memory=device.type == 'cuda')
    rows, diffs, alignment, checks, manifests, files, layouts, input_previews, result_previews = [],[],{},[],{},[],[],[],[]
    image_checks = {"source_modes":Counter(),"metadata_keys":Counter(),"alpha_range":[255,255],"captured_rgb_range":[1.0,0.0],"synthetic_rgb_range":[1.0,0.0],"frames":[]}
    with ThreadPoolExecutor(max_workers=3) as pool, torch.inference_mode():
        for index,batch in enumerate(loader):
            pair, old_row = dataset.pairs[index], captured_csv[index]
            hr = batch['hr'].to(device)
            cap_array = hwc(batch['lr'])
            for side,path in (("captured",pair.lr_path),("hr",pair.hr_path)):
                with Image.open(path) as image:
                    image_checks['source_modes'][side+':'+image.mode] += 1
                    image_checks['metadata_keys'].update(side+':'+key for key in image.info)
                    if 'A' in image.getbands():
                        lo,hi = image.getchannel('A').getextrema()
                        image_checks['alpha_range'] = [min(lo,image_checks['alpha_range'][0]),max(hi,image_checks['alpha_range'][1])]
                    if side == 'hr': hr_u8 = np.asarray(image.convert('RGB'))
            syn_u8 = synthetic_u8(hr_u8,resize)
            assert syn_u8.shape == (540,960,3)
            cache = synthetic_root/pair.hr_path.name
            Image.fromarray(syn_u8).save(cache,compress_level=3)
            with Image.open(cache) as image: decoded = np.array(image.convert('RGB'))
            assert np.array_equal(decoded,syn_u8)
            syn = torch.from_numpy(decoded.transpose(2,0,1).copy()).float().div(255).unsqueeze(0).to(device)
            assert syn.shape == batch['lr'].shape and syn.dtype == torch.float32 and torch.isfinite(syn).all() and syn.min() >= 0 and syn.max() <= 1
            if index == 0:
                reference_lr, reference_hr = official_pair(hr_u8,2)
                assert torch.equal(syn.cpu()[0],reference_lr) and torch.equal(batch['hr'][0],reference_hr)
                equivalence['first_real_frame_bitwise_equal'] = True
            syn_metrics, syn_arrays = predict(syn,hr,models,pool)
            syn_row = {"condition":"synthetic","name":pair.key,"lr_path":str(cache),"hr_path":str(pair.hr_path),**syn_metrics}
            cap_row = {"condition":"captured","name":pair.key,"lr_path":str(pair.lr_path),"hr_path":str(pair.hr_path)}
            cap_row.update({m+'_'+k:float(old_row[m+'_'+k]) for m in METHODS for k in ('psnr_db','ssim')})
            rows.extend((cap_row,syn_row))
            syn_array = hwc(syn)
            manifests[str(cache.relative_to(out))] = {**fingerprint(cache),"hr_name":pair.key,"wh":[960,540],"mode":"RGB","dtype":"uint8"}
            cap_tensor = batch['lr'].to(device)
            diff = {"name":pair.key,"lr_psnr_db":rgb_psnr_per_image(cap_tensor,syn).item(),"lr_ssim":rgb_ssim(cap_array,syn_array)}
            for name,array in (("captured",cap_array),("synthetic",syn_array)):
                gy,gx = np.gradient(array.mean(axis=2))
                diff[name+'_mean_squared_gray_gradient'] = float(np.mean(gx*gx+gy*gy))
                key = name+'_rgb_range'; image_checks[key] = [min(float(array.min()),image_checks[key][0]),max(float(array.max()),image_checks[key][1])]
            diffs.append(diff)
            if index in ROIS:
                cap_metrics, cap_arrays = predict(cap_tensor,hr,models,pool)
                deltas = {key:cap_metrics[key]-cap_row[key] for key in cap_metrics}
                assert all(abs(value) < (1e-4 if key.endswith('psnr_db') else 1e-6) for key,value in deltas.items()), deltas
                checks.append({"name":pair.key,"metric_deltas":deltas,"pass":True})
                alignment[pair.key] = alignment_probe(cap_array,syn_array,ROIS[index])
                sample_files,sample_layouts,previews = save_samples(out,index,cap_array,syn_array,hwc(hr),cap_arrays,syn_arrays,cap_row,syn_row)
                files.extend(sample_files); layouts.extend(sample_layouts)
                input_previews.append(previews['A_inputs']); result_previews.extend((previews['B_captured'],previews['B_synthetic']))
            image_checks['frames'].append({"name":pair.key,"lr_wh":[960,540],"hr_wh":[1920,1080],"synthetic_mode":"RGB"})
            if index == 0 or (index+1)%20 == 0:
                logger.info("Synthetic completed %d/240; fixed recipe; captured results reused",index+1)
    csv_write(out/"per_image_metrics.csv",rows)
    csv_write(out/"lr_difference_metrics.csv",diffs)
    comparison = []
    means = {condition:{method:{metric:sum(row[method+'_'+metric] for row in rows if row['condition']==condition)/240 for metric in ('psnr_db','ssim')} for method in METHODS} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for method in METHODS:
            score = means[condition][method]
            item = {"condition":condition,"method":method,"frames":240,"mean_psnr_db":score['psnr_db'],"mean_ssim":score['ssim']}
            for metric in ('psnr_db','ssim'):
                item['delta_'+metric+'_vs_bicubic'+('_db' if metric=='psnr_db' else '')] = score[metric]-means[condition]['bicubic'][metric]
                item['delta_'+metric+'_vs_original'+('_db' if metric=='psnr_db' else '')] = score[metric]-means[condition]['original'][metric]
                item['synthetic_minus_captured_'+metric] = means['synthetic'][method][metric]-means['captured'][method][metric]
            # Keep simple, conventional delta names for downstream tables.
            item['delta_psnr_vs_bicubic_db'] = item.pop('delta_psnr_db_vs_bicubic_db')
            item['delta_psnr_vs_original_db'] = item.pop('delta_psnr_db_vs_original_db')
            comparison.append(item)
    csv_write(out/"comparison_table.csv",comparison)
    previous.save_montage(input_previews,out/"samples/overview_inputs.png")
    previous.save_montage(result_previews,out/"samples/overview_reconstruction_rois.png")
    write_json(out/"figure_layouts.json",layouts)
    write_json(out/"synthetic_manifest.json",manifests)
    write_json(out/"alignment_checks.json",alignment)
    write_json(out/"data_checks.json",image_checks)
    write_json(out/"captured_sample_rechecks.json",checks)
    gains = {condition:{metric:means[condition]['finetuned'][metric]-means[condition]['original'][metric] for metric in ('psnr_db','ssim')} for condition in CONDITIONS}
    shifts = {method:{metric:means['synthetic'][method][metric]-means['captured'][method][metric] for metric in ('psnr_db','ssim')} for method in METHODS}
    observations = [f"{m}：Synthetic 相对 Captured 的 PSNR {shifts[m]['psnr_db']:+.6f} dB，SSIM {shifts[m]['ssim']:+.6f}。" for m in METHODS]
    observations += [f"微调相对 Original：Captured {gains['captured']['psnr_db']:+.6f} dB / SSIM {gains['captured']['ssim']:+.6f}；Synthetic {gains['synthetic']['psnr_db']:+.6f} dB / SSIM {gains['synthetic']['ssim']:+.6f}。"]
    wins = {c:{f'{a}_over_{b}':{metric:sum(row[a+'_'+metric] > row[b+'_'+metric] for row in rows if row['condition']==c) for metric in ('psnr_db','ssim')} for a,b in (('original','bicubic'),('finetuned','bicubic'),('finetuned','original'))} for c in CONDITIONS}
    original_recovers = shifts['original']['psnr_db']>0 and means['synthetic']['original']['psnr_db']>means['synthetic']['bicubic']['psnr_db']
    interpretation = ("在 HR 内容不变时，Original 的 Synthetic 分数提高且超过同条件 Bicubic，支持 LR 生成/采样方式不匹配是 Captured 条件性能下降的重要因素。" if original_recovers else "本次 Synthetic 结果需结合跨条件绝对变化与同条件 Bicubic 差值判断；没有达到‘提高且超过 Bicubic’的完整条件，不能直接宣称恢复正常预训练表现。")
    interpretation += "微调收益在两种输入下的变化用于判断其适应性集中在哪种条件，但不能将跨条件分差直接当作 Degradation Gap 的因果贡献。仍无法排除图像内容分布、残余对齐、AA/LOD/曝光/渲染后处理、下采样与训练配方细节、模型容量及验证集选模的影响。"
    after = snapshot(protected)
    write_json(out/"protected_files_after.json",after)
    assert before == after, "Protected source files changed during diagnostic"
    summary = {"status":"PASS","started_at_china_time":started,"completed_at_china_time":datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "dataset":captured_summary['dataset'],"checkpoints":captured_summary['checkpoints'],"model":captured_summary['model'],"evaluation":captured_summary['evaluation'],
        "captured_provenance":{"directory":str(captured_dir),"summary":fingerprint(captured_dir/"summary.json"),"per_image_metrics":fingerprint(captured_dir/"per_image_metrics.csv"),"full_results_reused":True,"sample_frames_recomputed":list(ROIS),"source_and_raw_input_hashes_match":True},
        "official_evidence":{"source_commit":COMMIT,"release_readme":OFFICIAL_URL+"QuickSRNet.md","evaluation_preprocessing":OFFICIAL_URL+"dataloader/utils.py","kernel_implementation":OFFICIAL_URL+"dataloader/imresize.py","paper":"https://arxiv.org/html/2303.04336","training_dataset_confirmed":"DIV2K","exact_training_degradation_confirmed":False,"public_evaluation_preprocessing_confirmed":True,"cache_checkpoint_crosscheck":cache_crosscheck,"limitation":"Public evaluation recipe is not proof of complete checkpoint training recipe"},
        "synthetic_generation":{"implementation":"Pinned official public evaluation imresize; executed from source snapshot","scale":0.5,"method":"bicubic","mode":"vec","kernel_a":-0.5,"kernel_width_base":4,"kernel_width_downsample":8,"antialias":True,"antialias_setting":"intrinsic scale*kernel(scale*x), no separate flag","coordinate_mapping":"half-pixel pixel-centre mapping","boundary":"symmetric repeated-edge indexing","input_dtype_range":"uint8 RGB -> float64 [0,255]","output_quantization":"np.uint8(np.clip(resized,0,255)): truncation, not rounding","model_input":"cached RGB PNG decoded -> BCHW float32 /255","hr_crop":None,"cached_images":240,"equivalence_checks":equivalence,"algorithm_selected_before_scores":True,"kernel_search_performed":False},
        "comparison":comparison,"diagnostic_deltas":{"synthetic_minus_captured":shifts,"finetuned_minus_original":gains},"per_frame_wins_out_of_240":wins,
        "lr_difference_means":{key:sum(row[key] for row in diffs)/240 for key in diffs[0] if key!='name'},"captured_sample_rechecks_passed":True,
        "alignment_assessment":"See alignment_checks.json; integer-shift probe only, no correction or proof of exact subpixel/temporal alignment",
        "software":{"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,"Pillow":PIL.__version__,"scikit_image":skimage.__version__,"scipy":scipy.__version__},
        "device":captured_summary['device'] | {"type":device.type,"name":torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'},
        "source_files":source_info,"command":[sys.executable,*sys.argv],"protected_file_count":len(before),"protected_files_unchanged":True,
        "test_set_accessed":False,"training_performed":False,"performance_benchmark_performed":False,"sample_files":files,"observations":observations,"interpretation":interpretation,"anomalies":[]}
    for key in ('torch','numpy','Pillow','scikit_image'):
        assert summary['software'][key] == captured_summary['software'][key], (key,'version mismatch with captured evaluator')
    write_json(out/"summary.json",summary)
    write_readme(out,summary)
    logger.info("PASS: 240 synthetic frames, captured metrics reused, protected files unchanged: %d",len(before))
    for item in comparison: logger.info("%s %s PSNR %.6f SSIM %.6f",item['condition'],item['method'],item['mean_psnr_db'],item['mean_ssim'])


if __name__ == "__main__":
    main()
