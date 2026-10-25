"""Compare Bicubic, official and epoch-91 QuickSRNet on Scene_1_02 only.

No training, test images, checkpoint writes, tiling, or speed benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import skimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset
from src.metrics import SSIM_PARAMETERS, rgb_psnr_per_image, rgb_ssim
from src.trainer import build_quicksrnet_small_from_local_checkpoint, resolve_device
from src.utils import create_logger, seed_everything

RUN = "asian-village-20261018-200240"
METHODS = ("bicubic", "original", "finetuned")
LABELS = ("LR (display only)", "Bicubic 2x", "Original QuickSRNet", "Fine-tuned (Epoch 91)", "HR Ground Truth")
# HR coordinates: x0,y0,x1,y1, zero-based, half-open; chosen from HR content before scoring.
ROIS = {
    0: (("A architectural edges", (1280, 240, 1536, 496)), ("B paving texture", (320, 760, 576, 1016))),
    60: (("A timber structure", (1280, 300, 1536, 556)), ("B paving texture", (600, 740, 856, 996))),
    120: (("A roof and facade", (700, 180, 956, 436)), ("B paving texture", (260, 740, 516, 996))),
    180: (("A roof and building", (660, 160, 916, 416)), ("B paving texture", (1250, 740, 1506, 996))),
    239: (("A timber facade", (470, 180, 726, 436)), ("B paving texture", (500, 740, 756, 996))),
}
ROI_COLORS = ("#f59e0b", "#06b6d4")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def snapshot(paths) -> dict:
    return {str(path.resolve()): fingerprint(path) for path in sorted(set(paths))}


def hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().permute(1, 2, 0).contiguous().numpy()


def image_from_float(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


def comparison_strip(images, metrics, title, width, rois=(), crop=None) -> Image.Image:
    gap = max(12, width // 64)
    scale = max(1.0, width / 512)
    heading_h, method_h = int(42 * scale), int(66 * scale)
    panels = []
    for source in images:
        panel = source.crop(crop) if crop is not None else source.copy()
        if crop is None:
            draw = ImageDraw.Draw(panel)
            for index, (_, box) in enumerate(rois):
                draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), outline=ROI_COLORS[index], width=4)
        if panel.width != width:
            height = round(panel.height * width / panel.width)
            interpolation = Image.Resampling.NEAREST if crop is not None else Image.Resampling.LANCZOS
            panel = panel.resize((width, height), interpolation)
        panels.append(panel)
    canvas = Image.new("RGB", (5 * width + 6 * gap, panels[0].height + heading_h + method_h + 2*gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), title, fill="#172033", font=font(int(21*scale), True))
    for index, (panel, label) in enumerate(zip(panels, LABELS)):
        left = gap + index * (width + gap)
        color = "#006b60" if index == 3 else "#172033"
        draw.text((left, heading_h), label, fill=color, font=font(int(20*scale), index == 3))
        if index in (1, 2, 3):
            method = METHODS[index-1]
            caption = f"PSNR {metrics[method+'_psnr_db']:.3f} dB | SSIM {metrics[method+'_ssim']:.4f}"
        elif index == 0:
            caption = "Nearest display; not scored"
        else:
            caption = "Reference RGB image"
        draw.text((left, heading_h + int(28*scale)), caption, fill="#526071", font=font(int(17*scale)))
        canvas.paste(panel, (left, heading_h + method_h))
    return canvas


def save_samples(out: Path, frame: int, lr: np.ndarray, outputs, hr: np.ndarray, row: dict):
    reference = image_from_float(hr)
    lr_display = image_from_float(lr).resize(reference.size, Image.Resampling.NEAREST)
    images = [lr_display] + [image_from_float(outputs[method]) for method in METHODS] + [reference]
    rois = ROIS[frame]
    for _, box in rois:
        assert 0 <= box[0] < box[2] <= reference.width and 0 <= box[1] < box[3] <= reference.height
    prefix = f"Scene_1_02_frame_{frame:04d}"
    full = comparison_strip(images, row, f"{prefix} | Native HR resolution | A: architecture (orange), B: texture (cyan)",
                            reference.width, rois=rois)
    full.save(out / f"{prefix}_full.png", compress_level=6, dpi=(150,150))
    previews = comparison_strip(images, row, f"{prefix} | Overview; detailed differences are in the aligned crops", 512, rois=rois)
    zoom_preview = None
    files = [f"samples/{prefix}_full.png"]
    for index, (label, box) in enumerate(rois):
        title = f"{prefix} | {label} | HR box {box} | 2x nearest zoom; scores above are full-frame"
        crop = comparison_strip(images, row, title, 512, crop=box)
        path = out / f"{prefix}_roi_{chr(65+index)}.png"
        crop.save(path, compress_level=6, dpi=(150,150))
        files.append(f"samples/{path.name}")
        if index == 0:
            zoom_preview = comparison_strip(images, row, f"{prefix} | {label} | Identical HR coordinates in every column", 256, crop=box)
    return previews, zoom_preview, files


def save_montage(rows, path: Path):
    gap = 20
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows) + gap*(len(rows)-1)), "#e7edf2")
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + gap
    canvas.save(path, compress_level=6, dpi=(150,150))


def write_readme(out: Path, summary: dict):
    table = ["| 方法 | 平均 RGB PSNR (dB) | 平均 RGB SSIM | ΔPSNR vs Original | ΔSSIM vs Original |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["comparison"]:
        table.append(f"| {row['method']} | {row['mean_psnr_db']:.6f} | {row['mean_ssim']:.6f} | "
                     f"{row['delta_psnr_vs_original_db']:+.6f} | {row['delta_ssim_vs_original']:+.6f} |")
    checks = summary["consistency_checks"]
    observations = "\n".join("- " + item for item in summary.get("observations", []))
    text = f'''# QuickSRNet-Small 2×：验证集初步评估

本次仅评估 Scene_1_02 的全部 240 对图像。未访问独立测试集 Scene_1_05，未训练或搜索模型超参数，也未进行推理速度 Benchmark。

## 结果

{chr(10).join(table)}

PSNR 差值的单位为 dB，SSIM 差值为绝对差值；二者都不等同于主观画质提升百分比。
逐帧提升比例见 summary.json 的 per_image_improvement 部分。完整逐帧数据和对 Bicubic 的差值见两个 CSV。

## 结果解读

{observations}

## 可复现命令

在 `F:\\CADCG\\SRTP\\QuickSRNet-Gaming` 中执行，输出目录应选择新的名称：

```powershell
conda activate qsr-gaming
python .\\scripts\\05_verify_validation.py --device cuda --output-dir outputs/verification/reproduce-val-scene1-02
```

实际本次输出：`{out}`。脚本默认权重路径与本次相同，并强制检查最佳权重来自总计 100 轮配置、实际保存 Epoch 91。

## 数据与模型

- LR：`{summary['dataset']['lr_root']}`。
- HR：`{summary['dataset']['hr_root']}`。
- 直接复用 PairedImageDataset：递归按相对路径匹配，PNG 转 RGB、除以 255，关闭随机裁剪和增强；240 对，LR 960×540、HR 1920×1080。
- 官方权重：`{summary['checkpoints']['official']['path']}`。
- 微调权重：`{summary['checkpoints']['finetuned']['path']}`，Epoch 91；没有加载旧的 50 轮权重或 Epoch 100 的 last.pth 来计算模型结果。
- 模型参数与 SHA-256、版本、设备、源码快照和后端设置见 summary.json。

## 统一评估约定

1. 三种方法都接收相同的 RGB [0,1] float32 LR，输出与整张 HR 严格对齐。模型使用 eval()、inference_mode()；FP32、batch size 1，不使用 AMP、分块或测试时增强。
2. Bicubic 使用 torch.nn.functional.interpolate，size=HR 的高宽、mode='bicubic'、align_corners=False、antialias=False。浮点插值结果统一 clamp 到 [0,1]，不经过 8-bit 图片量化再评分。
3. PSNR 复用 src.metrics.rgb_psnr_per_image；该函数由原 Trainer.validate 公式原样提取，已有逐值相等测试。完整 RGB 三通道、所有像素共同计算每张图 MSE，峰值为 1；不转 Y、不裁边、不添加 epsilon。
4. 所有逐帧 PSNR 取算术平均，即 mean(PSNR_i)，与已有训练验证聚合方式相同；没有用所有帧合并后的 MSE 替代。
5. SSIM 使用 scikit-image {summary['software']['scikit_image']} structural_similarity；HWC float32，data_range=1.0、channel_axis=-1、win_size=11、gaussian_weights=True、sigma=1.5、use_sample_covariance=False、K1=0.01、K2=0.03。
6. SSIM 三通道分别计算后等权平均，再对 240 张图片取算术平均。输入完整图片、不显式裁图；库内使用 reflect 高斯边界，并在 SSIM 图上剔除窗口半径 5 像素的边缘中心后求标准标量 SSIM。此约定对三种方法完全一致。PSNR 使用全体像素，不进行该窗口处理。
7. 指标均基于浮点预测。PNG 仅用于查看，统一将 [0,1] 四舍五入转 uint8，不用于反算指标；没有独立锐化、调色或增强。

## 历史结果一致性检查

- 官方参考：{checks['original']['reference_psnr_db']:.9f} dB；本次：{checks['original']['measured_psnr_db']:.9f} dB；差值：{checks['original']['difference_db']:+.9f} dB。
- 微调参考：{checks['finetuned']['reference_psnr_db']:.9f} dB；本次：{checks['finetuned']['measured_psnr_db']:.9f} dB；差值：{checks['finetuned']['difference_db']:+.9f} dB。
- 一致性容差为 0.001 dB；状态见 summary.json，未为对齐数字修改预处理或评分。
- 原始权重、现有训练日志与验证图片的 SHA-256、大小、mtime 在执行前后保持一致：{summary['protected_files_unchanged']}。

## 图像展示

列顺序固定为 LR（仅显示）、Bicubic、Original、Fine-tuned、HR。LR 使用最近邻放大到 HR 尺寸供展示，不参与评分。
样本固定为帧 0、60、120、180、239，先按序列位置选定，再按 HR 内容选择建筑和纹理区域；没有按指标增益筛选。选择计划记录在 sample_plan.json。
每张 native full 图保留完整 HR 分辨率，橙/青框在五列中位置相同；ROI 图使用相同 HR 坐标、256×256 裁剪，再统一最近邻放大 2 倍。
ROI 上方显示的是该帧整图指标，并非该 ROI 的单独指标。坐标均为从 0 开始的半开区间 [x0,y0,x1,y1)。总览缩放仅用于排版，查看细节应打开独立全图或局部 PNG。

- [五帧全图总览](samples/overview_full.png)
- [建筑局部总览](samples/overview_architecture.png)

## 文件

- [summary.json](summary.json)：平均指标、差值、数据/权重/模型/指标约定、完整性和历史一致性检查。
- [per_image_metrics.csv](per_image_metrics.csv)：240 行配对路径、尺寸、三种方法的 PSNR/SSIM 和逐帧差值。
- [comparison_table.csv](comparison_table.csv)：三种方法平均值及相对 Bicubic、Original 的指标差值。
- [sample_plan.json](sample_plan.json)：预先确定的帧号、选取规则和全部 ROI 坐标。
- samples/：五张完整五列图、十张局部五列图和两个总览。
- protected_files_before.json / protected_files_after.json：输入与现有成果的只读完整性记录。
- source/：本次评估和相关实现的源码快照；evaluation.log 记录进度，不写既有训练日志。

## 结果适用范围

这是已用于选择最佳模型的验证序列上的初步验证，不能代替独立测试结果或跨场景泛化结论。
PSNR 和 SSIM 衡量不同方面，应结合局部图观察边缘、纹理和可能的平滑；本次没有评估视频时域稳定性。
异常：{json.dumps(summary['anomalies'], ensure_ascii=False)}。
'''
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve()
    out.relative_to((ROOT / "outputs/verification").resolve())
    if any((out / name).exists() for name in ("summary.json", "per_image_metrics.csv", "evaluation.log")):
        raise FileExistsError("Use a fresh verification directory; existing results will not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    samples = out / "samples"
    samples.mkdir()
    china_timezone = timezone(timedelta(hours=8))
    started = datetime.now(china_timezone).isoformat()
    logger = create_logger("verification." + out.name, out / "evaluation.log")
    seed_everything(42)
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    lr_root, hr_root = ROOT / "data/gaming/val/LR", ROOT / "data/gaming/val/HR"
    for directory in (lr_root, hr_root):
        if sorted(p.name for p in directory.iterdir() if p.is_dir()) != ["Scene_1_02"]:
            raise ValueError(f"Only Scene_1_02 is allowed in validation: {directory}")
    official_path = ROOT / "checkpoints/pretrained/quicksrnet_small_2x_checkpoint_float32.pth.tar"
    fine_path = ROOT / "checkpoints/finetuned" / RUN / "best.pth"
    fine = torch.load(fine_path, map_location="cpu", weights_only=True)
    if (fine["epoch"] != 91 or fine["run_name"] != RUN or fine["scale_factor"] != 2
            or fine["config"]["training"]["epochs"] != 100 or not fine["validation_is_independent"]):
        raise ValueError("Expected the epoch-91 global best from the completed 100-epoch experiment")
    baseline_path = ROOT / "outputs/preflight-20261018/report.json"
    baseline_reference = json.loads(baseline_path.read_text(encoding="utf-8"))["pretrained_validation"]["psnr_db"]
    plan = {"selection": "Fixed temporal coverage before scoring; ROIs chosen by HR content, not gains",
            "frames": list(ROIS), "coordinate_system": "HR x0,y0,x1,y1; 0-based half-open",
            "rois": ROIS, "zoom": "256x256 HR crop enlarged 2x with nearest for every column"}
    write_json(out / "sample_plan.json", plan)
    dataset = PairedImageDataset(lr_root, hr_root, scale_factor=2, lr_patch_size=None, augment=False)
    if len(dataset) != 240 or any(pair.key.split('/')[0] != 'scene_1_02' for pair in dataset.pairs):
        raise ValueError("Expected exactly 240 Scene_1_02 pairs")
    if any(pair.lr_size != (960,540) or pair.hr_size != (1920,1080) for pair in dataset.pairs):
        raise ValueError("Unexpected image dimensions")
    protected = [pair.lr_path for pair in dataset.pairs] + [pair.hr_path for pair in dataset.pairs]
    protected += [official_path] + list(fine_path.parent.rglob("*.pth"))
    protected += [p for p in (ROOT / 'outputs/logs' / RUN).rglob('*') if p.is_file()]
    before = snapshot(protected)
    write_json(out / "protected_files_before.json", before)
    source = out / "source"
    source.mkdir()
    source_hashes = {}
    for path in (Path(__file__), ROOT/'src/metrics.py', ROOT/'src/dataset.py', ROOT/'src/trainer.py', ROOT/'tests/test_metrics.py'):
        destination = source / path.name
        shutil.copyfile(path, destination)
        source_hashes[str(path.relative_to(ROOT))] = fingerprint(path)["sha256"]
    logger.info("Validation only: %s | %s | pairs=%d", lr_root, hr_root, len(dataset))
    logger.info("Fine-tuned checkpoint: epoch=%d | reference PSNR=%.9f | %s", fine['epoch'], fine['best_val_psnr'], fine_path)
    models = {}
    for method in ("original", "finetuned"):
        model = build_quicksrnet_small_from_local_checkpoint(official_path, scale_factor=2)
        if method == "finetuned":
            model.model.load_state_dict(fine["state_dict"], strict=True)
        models[method] = model.float().to(device).requires_grad_(False).eval()
    assert sum(p.numel() for p in models['original'].parameters()) == 22860
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    rows, previews, zoom_previews, sample_files = [], [], [], []
    with ThreadPoolExecutor(max_workers=3) as pool, torch.inference_mode():
        for index, batch in enumerate(loader):
            lr, hr = batch['lr'].to(device), batch['hr'].to(device)
            predicted = {"bicubic": F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False, antialias=False)}
            predicted.update({method: model(lr) for method, model in models.items()})
            arrays, row = {}, {"name": batch['name'][0], "lr_path": str(dataset.pairs[index].lr_path),
                              "hr_path": str(dataset.pairs[index].hr_path),
                              "lr_width": lr.shape[3], "lr_height": lr.shape[2], "hr_width": hr.shape[3], "hr_height": hr.shape[2]}
            target_array = hwc(hr)
            for method in METHODS:
                output = predicted[method]
                if output.shape != hr.shape or not torch.isfinite(output).all():
                    raise RuntimeError(f"Invalid {method} output for {row['name']}")
                output = output.clamp(0,1)
                row[method+'_psnr_db'] = rgb_psnr_per_image(output, hr).item()
                arrays[method] = hwc(output)
            futures = {method: pool.submit(rgb_ssim, arrays[method], target_array) for method in METHODS}
            for method, future in futures.items():
                row[method+'_ssim'] = future.result()
                if not all(math.isfinite(row[method+'_'+metric]) for metric in ('psnr_db','ssim')):
                    raise RuntimeError(f"Non-finite metric for {row['name']}: {method}")
            for reference in ('bicubic', 'original'):
                for metric in ('psnr_db', 'ssim'):
                    row[f'delta_finetuned_minus_{reference}_{metric}'] = row['finetuned_'+metric] - row[reference+'_'+metric]
            rows.append(row)
            frame = int(Path(row['name']).name.split('_')[-1])
            if frame in ROIS:
                preview, zoom, files = save_samples(samples, frame, hwc(lr), arrays, target_array, row)
                previews.append(preview)
                zoom_previews.append(zoom)
                sample_files.extend(files)
            if (index+1) % 20 == 0 or index == 0:
                logger.info("Completed %d/240 | Original mean PSNR %.6f | Fine-tuned mean PSNR %.6f",
                            index+1, sum(r['original_psnr_db'] for r in rows)/len(rows), sum(r['finetuned_psnr_db'] for r in rows)/len(rows))
    if len(rows) != 240 or len({r['name'] for r in rows}) != 240:
        raise RuntimeError("Incomplete or duplicate evaluation rows")
    if len(previews) != 5:
        raise RuntimeError("All five preselected visualization frames must be present")
    with (out/'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {method: {metric: sum(r[method+'_'+metric] for r in rows)/len(rows) for metric in ('psnr_db','ssim')} for method in METHODS}
    comparison = []
    for method in METHODS:
        record = {'method': method, 'mean_psnr_db': means[method]['psnr_db'], 'mean_ssim': means[method]['ssim']}
        for reference in ('bicubic', 'original'):
            record['delta_psnr_vs_'+reference+'_db'] = means[method]['psnr_db'] - means[reference]['psnr_db']
            record['delta_ssim_vs_'+reference] = means[method]['ssim'] - means[reference]['ssim']
        comparison.append(record)
    with (out/'comparison_table.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save_montage(previews, samples/'overview_full.png')
    save_montage(zoom_previews, samples/'overview_architecture.png')
    checks, anomalies = {}, []
    for method, reference in (('original', baseline_reference), ('finetuned', fine['best_val_psnr'])):
        difference = means[method]['psnr_db'] - reference
        checks[method] = {'reference_psnr_db': reference, 'measured_psnr_db': means[method]['psnr_db'],
                          'difference_db': difference, 'tolerance_db': 0.001, 'pass': abs(difference) <= 0.001}
        if not checks[method]['pass']:
            anomalies.append(f'{method} historical PSNR difference {difference:+.9f} dB requires investigation')
    after = snapshot(protected)
    write_json(out/'protected_files_after.json', after)
    if before != after:
        anomalies.append('Protected input or existing training artifact changed during evaluation')
    improvement = {}
    for reference in ('bicubic', 'original'):
        improvement[reference] = {}
        for metric in ('psnr_db','ssim'):
            values = [r['finetuned_'+metric] - r[reference+'_'+metric] for r in rows]
            improvement[reference][metric] = {'improved_count': sum(v>0 for v in values), 'equal_count': sum(v==0 for v in values),
                'worse_count': sum(v<0 for v in values), 'improved_fraction': sum(v>0 for v in values)/len(values),
                'min_delta': min(values), 'max_delta': max(values)}
    summary = {
        'status': 'PASS' if not anomalies else 'REQUIRES_INVESTIGATION', 'started_at_china_time': started,
        'completed_at_china_time': datetime.now(china_timezone).isoformat(),
        'dataset': {'split': 'validation', 'sequence': 'Scene_1_02', 'pairs': len(rows), 'lr_root': str(lr_root), 'hr_root': str(hr_root),
                    'lr_wh': [960,540], 'hr_wh': [1920,1080], 'pairing': 'Existing PairedImageDataset relative stem matching'},
        'checkpoints': {'official': {'path': str(official_path), **before[str(official_path.resolve())]},
                        'finetuned': {'path': str(fine_path), 'epoch': fine['epoch'], 'training_target_epochs':100,
                                      'stored_best_val_psnr_db':fine['best_val_psnr'], **before[str(fine_path.resolve())]}},
        'model': {'name':'QuickSRNet-Small', 'scale_factor':2, 'channels':32, 'intermediate_layers':2,
                  'use_ito_connection':False, 'parameter_count':22860, 'architecture':str(models['original'])},
        'evaluation': {'color':'RGB', 'range':[0,1], 'dtype':'float32', 'batch_size':1, 'full_frame':True,
            'random_crop':False, 'augmentation':False, 'eval_mode':True, 'inference_mode':True, 'amp':False, 'tiling_used':False,
            'output_clamp':[0,1], 'metrics_before_uint8_export':True, 'psnr_function':'src.metrics.rgb_psnr_per_image',
            'psnr_aggregation':'arithmetic mean of per-image RGB PSNR in dB', 'psnr_border_shave':0,
            'ssim_function':'skimage.metrics.structural_similarity', 'ssim_parameters':SSIM_PARAMETERS,
            'ssim_aggregation':'equal RGB-channel mean per image, then arithmetic mean of images',
            'ssim_boundary':'Gaussian mode=reflect, truncate=3.5; scalar uses valid window centres, 5-pixel SSIM-map margin',
            'bicubic':{'library':'torch.nn.functional.interpolate', 'size_hw':[1080,1920], 'mode':'bicubic',
                       'align_corners':False, 'antialias':False, 'float_interpolation':True, 'clamp':[0,1]},
            'lr_display': 'Pillow nearest resize to HR size; display only, not scored', 'ssim_parallel_workers':3},
        'software': {'python':platform.python_version(), 'torch':str(torch.__version__), 'numpy':np.__version__,
                     'scikit_image':skimage.__version__, 'Pillow':PIL.__version__},
        'device': {'type':str(device), 'name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                   'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32, 'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_benchmark':torch.backends.cudnn.benchmark, 'cudnn_deterministic':torch.backends.cudnn.deterministic},
        'comparison': comparison, 'per_image_improvement':improvement, 'consistency_checks': checks,
        'reference_baseline_report':str(baseline_path), 'protected_file_count':len(before), 'protected_files_unchanged':before==after,
        'source_sha256':source_hashes, 'samples':sample_files, 'test_set_accessed':False, 'training_performed':False,
        'performance_benchmark_performed':False, 'anomalies':anomalies,
        'observations': [
            f"微调相对 Original 的平均 PSNR 增益为 {means['finetuned']['psnr_db']-means['original']['psnr_db']:.6f} dB，"
            f"相对 Bicubic 为 {means['finetuned']['psnr_db']-means['bicubic']['psnr_db']:.6f} dB。",
            f"微调相对 Original 在 {improvement['original']['psnr_db']['improved_count']}/240 帧 PSNR、"
            f"{improvement['original']['ssim']['improved_count']}/240 帧 SSIM 上改善；相对 Bicubic 分别为 "
            f"{improvement['bicubic']['psnr_db']['improved_count']}/240 和 {improvement['bicubic']['ssim']['improved_count']}/240。",
            "本序列中 Bicubic 高于官方模型时，应将其作为更强的基线一同报告；这与历史 PSNR 是否正确复现是两个问题，"
            "不能仅凭相对官方模型的增益判断对插值基线的实际收益。",
        ],
    }
    write_json(out/'summary.json', summary)
    write_readme(out, summary)
    logger.info('Completed: %s | all protected files unchanged=%s | %s', summary['status'], before==after, out)
    for record in comparison:
        logger.info('%s | mean RGB PSNR %.9f dB | mean RGB SSIM %.9f', record['method'], record['mean_psnr_db'], record['mean_ssim'])
    if anomalies:
        raise SystemExit('Evaluation requires investigation; see summary.json, without altering results to match references')


if __name__ == '__main__':
    main()
