"""One fixed-checkpoint Captured LR quality evaluation of Scene_1_05 (no training)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import skimage
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PairedImageDataset
from src.metrics import SSIM_PARAMETERS
from src.trainer import build_quicksrnet_small_from_local_checkpoint, resolve_device
from src.utils import create_logger, seed_everything


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load_module('prior_validation', ROOT/'scripts/05_verify_validation.py')
diagnostic = load_module('prior_diagnostic', ROOT/'scripts/06_diagnose_synthetic_lr.py')
# No synthetic generation is called: predict() is the prior unified, Captured-compatible evaluator.
predict = diagnostic.predict
fingerprint, snapshot, write_json = validation.fingerprint, validation.snapshot, validation.write_json
hwc, image_from_float = validation.hwc, validation.image_from_float
METHODS = ('bicubic','original','finetuned')
VAL = ROOT/'outputs/verification/val-scene1-02-20261019-140251'
DIAG = ROOT/'outputs/diagnostics/synthetic_lr/scene1-02-official-eval-20261019'
RUN = 'asian-village-20261018-200240'
# Fixed temporal sample frames; boxes chosen from HR only, before scoring.
# HR coordinates: xyxy, zero-based, half-open. Every box is 256x256 pixels.
ROIS = {
    0: (('A roof and architecture',(1120,160,1376,416)),('B vegetation',(40,280,296,536))),
    60: (('A roof and eaves',(1120,220,1376,476)),('B distant foliage',(280,420,536,676))),
    120: (('A roof and facade',(600,240,856,496)),('B character and edges',(860,420,1116,676))),
    180: (('A architectural details',(500,180,756,436)),('B stone wall texture',(1300,380,1556,636))),
    239: (('A distant pavilion',(980,280,1236,536)),('B vegetation and planter',(1200,440,1456,696))),
}
BAD_PSNR_DB = -0.5
BAD_SSIM = -0.01


def csv_write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_panels(out, frame, lr, arrays, hr, row, *, selected_reason=None):
    prefix = f'Scene_1_05_frame_{frame:04d}'
    target = image_from_float(hr)
    images = [image_from_float(lr).resize(target.size,Image.Resampling.NEAREST)]
    images += [image_from_float(arrays[m]) for m in METHODS]+[target]
    rois = ROIS.get(frame,()) if selected_reason is None else ()
    qualifier = 'Fixed temporal sample' if selected_reason is None else 'METRIC-SELECTED: '+selected_reason
    files, layouts = [],[]
    path = out/f'{prefix}_full.png'
    canvas = validation.comparison_strip(images,row,f'{prefix} | {qualifier} | Native full frame',1920,rois=rois)
    canvas.save(path,compress_level=3,dpi=(150,150))
    files.append(path)
    layouts.append({'file':str(path),'frame':frame,'selected_reason':selected_reason,'crop_hr_xyxy':None,'panel_width':1920,'gap':30,'panel_y':404,'panel_height':1080})
    preview = validation.comparison_strip(images,row,f'{prefix} | {qualifier}',512,rois=rois)
    zoom_preview = None
    for i,(label,box) in enumerate(rois):
        path = out/f'{prefix}_roi_{chr(65+i)}.png'
        canvas = validation.comparison_strip(images,row,f'{prefix} | {label} | HR ROI {box} | Full-frame scores',512,crop=box)
        canvas.save(path,compress_level=3,dpi=(150,150))
        files.append(path)
        layouts.append({'file':str(path),'frame':frame,'selected_reason':None,'crop_hr_xyxy':list(box),'panel_width':512,'gap':12,'panel_y':108,'panel_height':512})
        if i==0:
            zoom_preview = validation.comparison_strip(images,row,f'{prefix} | {label} | Identical HR crop',256,crop=box)
    return files, layouts, preview, zoom_preview


def improvement(rows):
    result = {}
    for reference in ('bicubic','original'):
        result[reference] = {}
        for metric in ('psnr_db','ssim'):
            deltas = np.array([r[f'delta_finetuned_minus_{reference}_{metric}'] for r in rows])
            result[reference][metric] = {
                'mean_delta':float(deltas.mean()),'improved_count':int((deltas>0).sum()),
                'equal_count':int((deltas==0).sum()),'worse_count':int((deltas<0).sum()),
                'min_delta':float(deltas.min()),'q05':float(np.quantile(deltas,.05)),
                'median_delta':float(np.median(deltas)),'q95':float(np.quantile(deltas,.95)),
                'max_delta':float(deltas.max()),
            }
    return result


def write_reports(out, summary):
    table = ['| 方法 | Test PSNR ↑ (dB) | Test SSIM ↑ | ΔPSNR vs Bicubic (dB) |','|---|---:|---:|---:|']
    labels = {'bicubic':'Bicubic 2×','original':'Original QuickSRNet-Small','finetuned':'Fine-tuned QuickSRNet-Small (Epoch 91)'}
    for item in summary['comparison']:
        delta = '—' if item['method']=='bicubic' else f"{item['delta_psnr_vs_bicubic_db']:+.6f}"
        table.append(f"| {labels[item['method']]} | {item['mean_psnr_db']:.6f} | {item['mean_ssim']:.6f} | {delta} |")
    gain = summary['per_image_improvement']
    narrative = []
    for ref in ('original','bicubic'):
        p,s = gain[ref]['psnr_db'],gain[ref]['ssim']
        narrative.append(f"微调相对 {labels[ref]}：平均 PSNR {p['mean_delta']:+.6f} dB，SSIM {s['mean_delta']:+.6f}；PSNR 在 {p['improved_count']}/240 帧更高、{p['worse_count']}/240 帧更低，SSIM 在 {s['improved_count']}/240 帧更高、{s['worse_count']}/240 帧更低。PSNR 最小/中位/最大差值为 {p['min_delta']:+.6f}/{p['median_delta']:+.6f}/{p['max_delta']:+.6f} dB。")
    validation_table = ['| 方法 | Scene_1_02 Val PSNR / SSIM | Scene_1_05 Test PSNR / SSIM |','|---|---:|---:|']
    for row in summary['validation_comparison']:
        validation_table.append(f"| {labels[row['method']]} | {row['val_psnr_db']:.6f} / {row['val_ssim']:.6f} | {row['test_psnr_db']:.6f} / {row['test_ssim']:.6f} |")
    positive = all(gain[r]['psnr_db']['mean_delta']>0 and gain[r]['ssim']['mean_delta']>0 for r in ('bicubic','original'))
    q1 = '是，平均 PSNR 与 SSIM 均提高。' if all(gain['original'][m]['mean_delta']>0 for m in ('psnr_db','ssim')) else '不能给出两项均提高的结论，具体变化见表。'
    q2 = '是，平均 PSNR 与 SSIM 均超过 Bicubic。' if all(gain['bicubic'][m]['mean_delta']>0 for m in ('psnr_db','ssim')) else '不能给出两项均超过的结论，具体变化见表。'
    q3 = '微调在两个序列上均优于两个基线，方向一致。' if positive else '与验证集的提升方向存在差异，需按本次固定结果如实报告。'
    q4 = ('结果支持本次微调缓解了当前 Asian Village 游戏渲染数据的分布不匹配，并能迁移到未参与训练和选模的同环境序列。' if positive else '当前测试结果不足以支持稳定缓解分布不匹配的结论。')+'它不能分解图像内容差异与退化/采样差异的因果贡献，也不能证明其他游戏或引擎上的泛化。'
    report = f"""# QuickSRNet-Small 2× 独立测试结果（SRTP 阶段报告）

## 实验目的与固定设置

在 Asian Village 的未参与训练和选模的 Scene_1_05 序列上，对固定的官方预训练模型、100 Epoch 实验验证最优的 Epoch 91 微调模型及 Bicubic 进行质量评估。使用全部 240 对真实采集 Captured LR（960×540）与 HR（1920×1080），未生成或使用 Synthetic LR。模型选择在本次测试前完成。

## 核心结果

{chr(10).join(table)}

{chr(10).join('- '+p for p in narrative)}

结果为逐帧 RGB PSNR（dB）和 RGB SSIM 的算术平均，未裁剪 PSNR 边界，所有方法共用同一配对及后处理。PSNR 差值不是主观画质百分比。

## 与此前验证结果的关系

{chr(10).join(validation_table)}

{q3} 两张表来自**不同序列**，其纹理、视角、角色占比和细节难度不同，不能将原始 PSNR 数值的升降直接归因为泛化能力变化。应结合各自同条件基线的差值来判断。验证集用于选择 Epoch 91，测试集用于这次固定模型的最终独立评估；连续 240 帧不能视为 240 个独立场景，不据此报告独立样本显著性。

## 研究问题

- Q1：微调是否优于官方模型？{q1}
- Q2：微调是否超过 Bicubic？{q2}
- Q3：与验证结论是否一致？{q3}
- Q4：是否缓解当前游戏渲染数据上的不匹配？{q4}
- Q5：伪影、结构损失与困难内容：见本文末尾的人工视觉检查补充。指标本身不能排除局部细节损失，也未评价时间稳定性。

## 较差帧检查与局限

评分前固定“明显低于基线”阈值为任一基线 ΔPSNR < {BAD_PSNR_DB} dB 或 ΔSSIM < {BAD_SSIM}；匹配帧 {len(summary['below_baseline_frames'])} 个。所有低于基线的帧与完整差值分布均保留在 summary.json / per_image_metrics.csv。

5 个主要展示帧（0000/0060/0120/0180/0239）与 ROI 在评分前根据 HR 内容固定，涵盖建筑边缘、植被、纹理、远处细节和角色边缘。额外保留“对 Bicubic PSNR 增益最小”和“微调绝对 PSNR 最低”的全图，明确放在 samples/metric_selected/；它们是**根据测试指标挑选的诊断样例**，不能冒充随机或代表性样本，且未用于模型选择或调整。

早期预检报告曾读取测试图像来核对配对和尺寸，没有测试质量指标。现有训练入口、配置、日志和 checkpoint 历史仅使用 train/val；本次也核验三划分路径与文件哈希不交叉。该证据支持未以测试集训练或选模；同一游戏中的共享资产、相近视角以及缺少完整外部访问审计仍是独立性边界。

这是一条同游戏环境的独立序列，不能推广为跨游戏或跨引擎结果。Synthetic LR 的历史诊断仅作背景，不与本次 Captured 测试混算，也不把对其性能的恢复直接当作本次泛化证据。
"""
    (out/'test_report.md').write_text(report,encoding='utf-8')
    readme = f"""# Scene_1_05 正式 Captured LR 测试

{chr(10).join(table)}

## 运行

从项目根目录用固定环境运行。已有输出目录不会覆盖；若默认日期目录已存在，自动创建时间戳后缀目录。

```powershell
& 'D:/anaconda3/envs/qsr-gaming/python.exe' scripts/07_evaluate_test.py --device cuda --output-dir outputs/test/scene1-05-20261019
```

## Pipeline

1. 检查配置、训练日志、完整 100 Epoch 的 metrics.csv、官方和微调 checkpoint 元数据与 SHA256；核对 best.pth 为 Epoch 91 且为验证集全程最优。检查旧验证与 Synthetic 报告的共享代码/权重/参数一致性。
2. 使用现有 PairedImageDataset(data/gaming/test/LR, data/gaming/test/HR,scale_factor=2,lr_patch_size=None,augment=False)，严格要求仅 Scene_1_05、frame_0000～frame_0239、240 对，尺寸分别 960×540 / 1920×1080。
3. 复用 scripts/06_diagnose_synthetic_lr.py 的 predict() 和 src.metrics 函数。该函数只做输入 LR 的统一推理/评分；本次没有调用任何 Synthetic LR 生成代码。
4. 两个模型用既有构造器和 strict state_dict 加载。eval()、requires_grad_(False)、torch.inference_mode()，batch=1、float32，无 AMP、TTA、分块、额外锐化/降噪或其他后处理。输出 clamp [0,1]，先评分后量化导出 PNG。
5. 导出逐帧与总体表、固定帧全图及相同 ROI；额外记录按指标选取的困难样例，随后进行独立 CSV/JSON/PNG/文件完整性核对。

## 评分约定

- 图像使用 Pillow convert('RGB')，CHW float32 /255；不额外做 gamma/ICC 变换。测试 PNG 模式、alpha、元数据统计见 data_checks.json。
- PSNR：src.metrics.rgb_psnr_per_image，RGB 等权、peak=1、整图，无边缘裁剪或 epsilon；每帧转 dB，再取 240 帧算术平均。
- SSIM：src.metrics.rgb_ssim → skimage.metrics.structural_similarity，参数 {SSIM_PARAMETERS}。RGB 等权后逐帧等权。Gaussian mode=reflect、truncate=3.5；标量平均有效窗口中心（11×11 窗在 SSIM map 边缘排除 5 像素），输入图本身不裁剪。
- Bicubic：torch.nn.functional.interpolate(size=(1080,1920),mode='bicubic',align_corners=False,antialias=False)，然后 clamp [0,1]，与原验证完全一致。
- 软件版本、实际模型配置、权重完整路径与 SHA256、实际命令、设备数学选项见 summary.json。

## 文件与解读

- [SRTP 测试报告](test_report.md)：结果、增益、验证集比较及 Q1–Q5。
- [summary.json](summary.json)：配置、均值、差值、逐帧胜负与极值、独立性证据。
- [comparison_table.csv](comparison_table.csv) / [per_image_metrics.csv](per_image_metrics.csv)：三方法总表与 240 帧明细。
- [固定帧全图概览](samples/overview_full.png) / [固定帧 ROI 概览](samples/overview_roi.png)。全分辨率 PNG 在 samples/，图片上方指标均为整帧值。LR 仅 nearest 2× 展示，不计入三方法评分；各方法 ROI 使用相同 HR 坐标和 nearest 2× 放大。
- sample_plan.json：评分前固定的帧、ROI、明显较差阈值；figure_layouts.json：可复核的像素版式。
- independence_audit.json / dataset_manifest.json：划分、文件哈希与历史配置/日志检查；检查重复文件不等于证明跨场景语义独立。
- protected_files_before.json / after.json：原始图像、权重、配置、训练曲线、日志、全部历史输出与源码的哈希、大小、mtime；本次前后一致。
- source/：本次脚本与共享实现快照。独立审计脚本和结果为 check_outputs.py / quality_checks.json。

本次不更新参数、不根据测试结果选择 Epoch 或搜索超参数，不计算推理延迟 Benchmark。测试结果是固定模型记录；后续优化须另行设计验证实验，不能据此反复调模型。
"""
    (out/'README.md').write_text(readme,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/test/scene1-05-20261019'))
    parser.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    args = parser.parse_args()
    out = (ROOT/args.output_dir).resolve(); out.relative_to(ROOT/'outputs/test')
    if out.exists():
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        out = out.with_name(out.name+'-'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    (out/'samples/metric_selected').mkdir(parents=True); (out/'source').mkdir()
    logger = create_logger('test.'+out.name,out/'evaluation.log')
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    logger.info('Fixed Scene_1_05 Captured LR evaluation; output=%s',out)
    val_summary = json.loads((VAL/'summary.json').read_text(encoding='utf-8'))
    diag_summary = json.loads((DIAG/'summary.json').read_text(encoding='utf-8'))
    assert val_summary['status']==diag_summary['status']=='PASS'
    assert not val_summary['test_set_accessed'] and not diag_summary['test_set_accessed']
    assert val_summary['evaluation']==diag_summary['evaluation']
    assert val_summary['evaluation']['ssim_parameters']==SSIM_PARAMETERS
    assert val_summary['evaluation']['bicubic']=={'library':'torch.nn.functional.interpolate','size_hw':[1080,1920],'mode':'bicubic','align_corners':False,'antialias':False,'float_interpolation':True,'clamp':[0,1]}
    config = yaml.safe_load((ROOT/'configs/asian_village_100.yaml').read_text(encoding='utf-8'))
    official = ROOT/'checkpoints/pretrained/quicksrnet_small_2x_checkpoint_float32.pth.tar'
    best = ROOT/'checkpoints/finetuned'/RUN/'best.pth'
    fine = torch.load(best,map_location='cpu',weights_only=True)
    assert fine['epoch']==91 and fine['config']['training']['epochs']==100 and fine['scale_factor']==2 and fine['run_name']==RUN
    assert fine['config']==config and fine['validation_is_independent']
    assert config['training']['optimizer']=='adam' and config['training']['loss']=='l1'
    for key in ('train_lr','train_hr','val_lr','val_hr'):
        assert config['data'][key] == 'data/gaming/'+('train/' if key.startswith('train') else 'val/')+('LR' if key.endswith('lr') else 'HR')
    train_log = (ROOT/'outputs/logs'/RUN/'train.log').read_text(encoding='utf-8')
    assert 'Train pairs: 960 | validation pairs: 240' in train_log
    assert 'data\\gaming\\test' not in train_log.lower() and 'scene_1_05' not in train_log.lower()
    assert 'data\\gaming\\train\\LR' in train_log and 'data\\gaming\\val\\LR' in train_log
    with (ROOT/'outputs/metrics'/RUN/'metrics.csv').open(encoding='utf-8-sig',newline='') as handle:
        history = list(csv.DictReader(handle))
    assert [int(r['epoch']) for r in history]==list(range(1,101))
    winner = max(history,key=lambda row:float(row['val_psnr_db']))
    assert int(winner['epoch'])==91 and float(winner['val_psnr_db'])==fine['best_val_psnr']
    for name,path in (('official',official),('finetuned',best)):
        assert fingerprint(path)['sha256']==val_summary['checkpoints'][name]['sha256']==diag_summary['checkpoints'][name]['sha256']
    for source in ('src/metrics.py','src/dataset.py','src/trainer.py','scripts/05_verify_validation.py'):
        assert fingerprint(ROOT/source)['sha256']==val_summary['source_sha256'][str(Path(source))]
    assert fingerprint(ROOT/'scripts/06_diagnose_synthetic_lr.py')['sha256']==diag_summary['source_files'][str(ROOT/'scripts/06_diagnose_synthetic_lr.py')]['sha256']
    roots = {split:{side:ROOT/'data/gaming'/split/side for side in ('LR','HR')} for split in ('train','val','test')}
    expected_scenes = {'train':['Scene_1_01','Scene_1_03','Scene_1_04','Scene_1_06'],'val':['Scene_1_02'],'test':['Scene_1_05']}
    inventories = {}
    for split,sides in roots.items():
        inventories[split] = {}
        for side,path in sides.items():
            assert sorted(p.name for p in path.iterdir() if p.is_dir())==expected_scenes[split]
            images = sorted(path.rglob('*.png'))
            assert len(images)==(960 if split=='train' else 240)
            inventories[split][side] = images
    test = PairedImageDataset(roots['test']['LR'],roots['test']['HR'],scale_factor=2,lr_patch_size=None,augment=False)
    assert len(test)==240 and [p.key for p in test.pairs]==[f'scene_1_05/frame_{i:04d}' for i in range(240)]
    assert all(p.lr_size==(960,540) and p.hr_size==(1920,1080) for p in test.pairs)
    raw_paths = [p for split in inventories.values() for files in split.values() for p in files]
    protected = list(raw_paths)
    for folder in ('checkpoints','configs'):
        protected += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    protected += [p for p in (ROOT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    sources = [Path(__file__),ROOT/'scripts/05_verify_validation.py',ROOT/'scripts/06_diagnose_synthetic_lr.py',ROOT/'scripts/03_finetune.py',ROOT/'src/dataset.py',ROOT/'src/metrics.py',ROOT/'src/trainer.py',ROOT/'src/utils.py']
    protected += sources
    before = snapshot(protected)
    write_json(out/'protected_files_before.json',before)
    source_info = {}
    for path in sources:
        dest = out/'source'/path.name; shutil.copyfile(path,dest)
        source_info[str(path)] = {**before[str(path)],'snapshot':str(dest.relative_to(out))}
    manifest = []
    hashes_by_split = {}
    for split,sides in inventories.items():
        hashes_by_split[split] = set()
        for side,files in sides.items():
            for path in files:
                item = {'split':split,'side':side,'path':str(path),'relative_stem':str(path.relative_to(roots[split][side]).with_suffix('')),**before[str(path)]}
                manifest.append(item); hashes_by_split[split].add(item['sha256'])
    overlaps = {f'{a}_vs_{b}':sorted(hashes_by_split[a]&hashes_by_split[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    assert not any(overlaps.values()),'Exact duplicate files across splits; do not silently score as independent'
    preflight = json.loads((ROOT/'outputs/preflight-20261018/report.json').read_text(encoding='utf-8'))
    assert preflight['splits']['test']['sequences']=={'scene_1_05':240} and 'pretrained_test' not in preflight
    independence = {'train_scenes':expected_scenes['train'],'val_scenes':expected_scenes['val'],'test_scenes':expected_scenes['test'],
        'train_pairs':960,'val_pairs':240,'test_pairs':240,'data_roots_from_checkpoint':config['data'],
        'training_entrypoint':'scripts/03_finetune.py uses only train_dataset and val_dataset; fit(train_loader,val_loader)',
        'training_log_roots_checked':True,'full_100_epoch_validation_history_best_epoch':int(winner['epoch']),
        'test_quality_not_used_for_checkpoint_selection':True,'historical_validation_test_set_accessed':val_summary['test_set_accessed'],
        'historical_diagnostic_test_set_accessed':diag_summary['test_set_accessed'],
        'early_preflight_test_pair_and_dimension_inspection':True,'early_preflight_test_quality_metrics_present':False,
        'exact_file_sha256_overlap':overlaps,'test_used_for_training_or_tuning_this_run':False,
        'scope_limit':'Audit is based on available config/code/logs/history and exact-file hashes; it does not prove absence of external usage, semantic overlap or shared assets in the same game.'}
    write_json(out/'dataset_manifest.json',manifest); write_json(out/'independence_audit.json',independence)
    plan = {'fixed_before_scoring':True,'selection':'Temporal coverage; ROIs chosen from HR content, not test gains','frames':list(ROIS),'rois':ROIS,
        'coordinates':'HR xyxy, zero-based half-open','display':'LR nearest to HR; all ROI columns 256x256 enlarged 2x nearest',
        'clearly_below_baseline_threshold':{'delta_psnr_db_less_than':BAD_PSNR_DB,'delta_ssim_less_than':BAD_SSIM,'either_baseline':True},
        'additional_metric_selected_examples':['minimum fine-tuned minus bicubic PSNR','minimum absolute fine-tuned PSNR'],
        'test_results_used_to_adjust_model':False}
    write_json(out/'sample_plan.json',plan)
    seed_everything(42); torch.set_num_threads(4); device = resolve_device(args.device)
    models = {}
    for name in ('original','finetuned'):
        model = build_quicksrnet_small_from_local_checkpoint(official,scale_factor=2)
        if name=='finetuned': model.model.load_state_dict(fine['state_dict'],strict=True)
        models[name] = model.float().to(device).requires_grad_(False).eval()
    assert all(sum(p.numel() for p in model.parameters())==22860 for model in models.values())
    loader = DataLoader(test,batch_size=1,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    rows, full_previews, zoom_previews, files, layouts = [],[],[],[],[]
    weakest = {}
    metadata = {'source_modes':Counter(),'metadata_keys':Counter(),'alpha_range':[255,255],'lr_range':[1.,0.],'hr_range':[1.,0.]}
    logger.info('Prechecks passed: pairs=240; LR=Captured; best epoch=91; checkpoint hashes match; exact-file split overlap=0')
    with ThreadPoolExecutor(max_workers=3) as pool, torch.inference_mode():
        for index,batch in enumerate(loader):
            pair = test.pairs[index]
            lr,hr = batch['lr'].to(device),batch['hr'].to(device)
            assert torch.isfinite(lr).all() and torch.isfinite(hr).all() and lr.min()>=0 and lr.max()<=1 and hr.min()>=0 and hr.max()<=1
            metrics,arrays = predict(lr,hr,models,pool)
            row = {'name':pair.key,'lr_path':str(pair.lr_path),'hr_path':str(pair.hr_path),'lr_source':'captured',
                   'lr_width':960,'lr_height':540,'hr_width':1920,'hr_height':1080,**metrics}
            for ref in ('bicubic','original'):
                for metric in ('psnr_db','ssim'):
                    row[f'delta_finetuned_minus_{ref}_{metric}'] = row['finetuned_'+metric]-row[ref+'_'+metric]
            rows.append(row)
            for side,path,tensor in (('lr',pair.lr_path,lr),('hr',pair.hr_path,hr)):
                with Image.open(path) as image:
                    metadata['source_modes'][side+':'+image.mode] += 1
                    metadata['metadata_keys'].update(side+':'+key for key in image.info)
                    if 'A' in image.getbands():
                        lo,hi = image.getchannel('A').getextrema()
                        metadata['alpha_range']=[min(lo,metadata['alpha_range'][0]),max(hi,metadata['alpha_range'][1])]
                metadata[side+'_range']=[min(float(tensor.min()),metadata[side+'_range'][0]),max(float(tensor.max()),metadata[side+'_range'][1])]
            for reason,key in (('minimum_gain_vs_bicubic','delta_finetuned_minus_bicubic_psnr_db'),('lowest_finetuned_psnr','finetuned_psnr_db')):
                if reason not in weakest or row[key]<weakest[reason]['row'][key]:
                    weakest[reason]={'frame':index,'row':row.copy(),'lr':hwc(lr).copy(),'hr':hwc(hr).copy(),'arrays':{m:a.copy() for m,a in arrays.items()}}
            if index in ROIS:
                f,l,preview,zoom = save_panels(out/'samples',index,hwc(lr),arrays,hwc(hr),row)
                files.extend(f); layouts.extend(l); full_previews.append(preview); zoom_previews.append(zoom)
            if index==0 or (index+1)%20==0:
                logger.info('Scored %d/240 test frames; fixed checkpoint and evaluation protocol',index+1)
    assert len(rows)==240 and len({r['name'] for r in rows})==240
    selected_info = []
    for reason,record in weakest.items():
        directory = out/'samples/metric_selected'/reason; directory.mkdir()
        f,l,_,_ = save_panels(directory,record['frame'],record['lr'],record['arrays'],record['hr'],record['row'],selected_reason=reason)
        files.extend(f); layouts.extend(l)
        selected_info.append({'reason':reason,'selection':'Based on test metrics; diagnostic only, not a representative/random sample','frame':record['frame'],'metrics':record['row'],'files':[str(p.relative_to(out)) for p in f]})
    csv_write(out/'per_image_metrics.csv',rows)
    comparison = []
    for method in METHODS:
        item = {'method':method,'frames':240,'mean_psnr_db':sum(r[method+'_psnr_db'] for r in rows)/240,'mean_ssim':sum(r[method+'_ssim'] for r in rows)/240}
        for ref in ('bicubic','original'):
            item[f'delta_psnr_vs_{ref}_db']=item['mean_psnr_db']-sum(r[ref+'_psnr_db'] for r in rows)/240
            item[f'delta_ssim_vs_{ref}']=item['mean_ssim']-sum(r[ref+'_ssim'] for r in rows)/240
        comparison.append(item)
    csv_write(out/'comparison_table.csv',comparison)
    validation.save_montage(full_previews,out/'samples/overview_full.png')
    validation.save_montage(zoom_previews,out/'samples/overview_roi.png')
    for layout in layouts: layout['file']=str(Path(layout['file']).relative_to(out))
    write_json(out/'figure_layouts.json',layouts); write_json(out/'data_checks.json',metadata)
    gains = improvement(rows)
    bad = [r for r in rows if any(r[f'delta_finetuned_minus_{ref}_psnr_db']<BAD_PSNR_DB or r[f'delta_finetuned_minus_{ref}_ssim']<BAD_SSIM for ref in ('bicubic','original'))]
    validation_comparison = []
    for v,t in zip(val_summary['comparison'],comparison):
        assert v['method']==t['method']
        validation_comparison.append({'method':t['method'],'val_psnr_db':v['mean_psnr_db'],'val_ssim':v['mean_ssim'],'test_psnr_db':t['mean_psnr_db'],'test_ssim':t['mean_ssim'],
            'val_delta_psnr_vs_bicubic_db':v['delta_psnr_vs_bicubic_db'],'test_delta_psnr_vs_bicubic_db':t['delta_psnr_vs_bicubic_db']})
    after = snapshot(protected); write_json(out/'protected_files_after.json',after)
    assert before==after,'Protected files changed'
    software = {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scikit_image':skimage.__version__,'Pillow':PIL.__version__}
    assert software==val_summary['software'],'Software differs from prior scoring'
    summary = {'status':'PASS','started_at_china_time':started,'completed_at_china_time':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'dataset':{'split':'test','sequence':'Scene_1_05','pairs':240,'lr_source':'captured','lr_root':str(roots['test']['LR']),'hr_root':str(roots['test']['HR']),'lr_wh':[960,540],'hr_wh':[1920,1080],'pairing':'Existing PairedImageDataset relative stem matching'},
        'checkpoints':val_summary['checkpoints'],'model':val_summary['model'],'training_config':config,'evaluation':val_summary['evaluation'],
        'comparison':comparison,'per_image_improvement':gains,'below_baseline_frames':bad,'clearly_below_baseline_threshold':plan['clearly_below_baseline_threshold'],
        'metric_selected_examples':selected_info,'validation_comparison':validation_comparison,'independence_audit':independence,
        'historical_results':{str(p):fingerprint(p) for p in (VAL/'summary.json',DIAG/'summary.json',DIAG/'README.md')},
        'software':software,'device':{'type':device.type,'name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU','cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic},
        'source_files':source_info,'command':[sys.executable,*sys.argv],'protected_file_count':len(before),'protected_files_unchanged':True,
        'sample_files':[str(p.relative_to(out)) for p in files],'training_performed':False,'synthetic_lr_used':False,'test_based_model_selection':False,'hyperparameter_search_performed':False,'performance_benchmark_performed':False,
        'anomalies':[],'visual_review_status':'pending; append after viewing fixed and metric-selected figures',
        'scope':'One unseen sequence within Asian Village; not cross-game or cross-engine generalization. Continuous frames are correlated.'}
    write_json(out/'summary.json',summary); write_reports(out,summary)
    logger.info('PASS: 240 test frames; protected files unchanged=%d; no training or checkpoint updates',len(before))
    for row in comparison: logger.info('%s PSNR %.9f SSIM %.9f',row['method'],row['mean_psnr_db'],row['mean_ssim'])


if __name__=='__main__':
    main()
