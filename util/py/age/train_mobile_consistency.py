"""Fine-tune the existing age regressor against small face-crop perturbations.

Keeps architecture/output decoding unchanged. Validation/test stay deterministic;
the user's diagnostic photo is never training data. Run in a separate output dir.
"""
import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from ab.nn.loader.utkface import loader
from ab.nn.nn.MobileAgeNet import Net

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
# Held fixed for validation, never sampled during model selection.
VARIANTS = ((1., 0., 0.), (.95, 0., 0.), (1.05, 0., 0.),
            (1., -.02, 0.), (1., .02, 0.), (1., 0., -.02), (1., 0., .02))


def crop_variant(image, scale=1., dx=0., dy=0.):
    image = image.convert('RGB')
    if (scale, dx, dy) == (1., 0., 0.):
        return image
    w, h = image.size
    px, py = round(w * .15), round(h * .15)
    padded = Image.fromarray(np.pad(np.asarray(image), ((py, py), (px, px), (0, 0)), mode='reflect'))
    cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
    left = round(px + (w - cw) / 2 + dx * w)
    top = round(py + (h - ch) / 2 + dy * h)
    if left < 0 or top < 0 or left + cw > padded.width or top + ch > padded.height:
        raise ValueError('Crop perturbation exceeds reflection padding')
    return padded.crop((left, top, left + cw, top + ch))


def tensor(image):
    return TF.normalize(TF.to_tensor(image.resize((224, 224), Image.Resampling.BILINEAR)), MEAN, STD)


class PairedCrops(Dataset):
    def __init__(self, split, training=False):
        self.split, self.training = split, training

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split.dataset.dataset[self.split.indices[index]]
        image = item['image'].convert('RGB')
        label = torch.tensor(float(item['age']), dtype=torch.float32)
        if self.training:
            if random.random() < .5:
                image = TF.hflip(image)
            altered = crop_variant(image, random.uniform(.95, 1.05),
                                   random.uniform(-.02, .02), random.uniform(-.02, .02))
            return tensor(image), tensor(altered), label
        return torch.stack([tensor(crop_variant(image, *v)) for v in VARIANTS]), label


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    errors, ranges = [], []
    for inputs, labels in data:
        b, v, c, h, w = inputs.shape
        outputs = model(inputs.reshape(b * v, c, h, w).to(device)).reshape(b, v).cpu()
        errors.extend((outputs[:, 0] - labels).abs().tolist())
        ranges.extend((outputs.max(1).values - outputs.min(1).values).tolist())
    return {'n': len(errors), 'mae': float(np.mean(errors)),
            'mean_crop_range': float(np.mean(ranges)), 'p95_crop_range': float(np.percentile(ranges, 95))}


def eligible(candidate, baseline):
    return (candidate['mae'] <= baseline['mae'] + .25
            and candidate['mean_crop_range'] < baseline['mean_crop_range']
            and candidate['p95_crop_range'] < baseline['p95_crop_range'])


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--consistency-weight', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not np.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        raise ValueError('Consistency weight must be finite and nonnegative')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing experiments are never overwritten')
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU job has no CUDA device')
    parameters = dict(pretrained=0, dropout=.1807417434748134, bounded_output=1,
                      min_age=0., max_age=116., train_min_age=1., train_max_age=95.)
    model = Net((1, 3, 224, 224), (1,), parameters, device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    for name, implementation in (('model_source.py', Net), ('loader_source.py', loader)):
        (args.output / name).write_text(Path(inspect.getfile(implementation)).read_text())
    save_json(args.output / 'config.json', {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'model_parameters': parameters, 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source_sha256': hashlib.sha256(Path(inspect.getfile(Net)).read_bytes()).hexdigest(),
        'variants': VARIANTS, 'consistency_weight': args.consistency_weight, 'torch': str(torch.__version__),
        'normalization': {'mean': MEAN, 'std': STD},
        'resize': 'RGB PIL bilinear 224x224', 'selection': 'validation only; MAE within 0.25 years of baseline and improved mean/p95 crop range',
        'batchnorm_and_dropout': 'eval mode for stable paired-view targets; gradients remain enabled'})
    _, _, train, val = loader(lambda _: tensor, 'age-regression')
    def batches(split, training=False):
        return DataLoader(PairedCrops(split, training), batch_size=args.batch_size,
                          shuffle=training, num_workers=args.workers, worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == 'cuda')
    training, validation = batches(train, True), batches(val)
    baseline = evaluate(model, validation, device)
    save_json(args.output / 'baseline_validation.json', baseline)
    print('Baseline:', json.dumps(baseline), flush=True)
    optimizer = torch.optim.AdamW([{'params': model.features.parameters(), 'lr': 1e-5},
                                  {'params': model.head.parameters(), 'lr': 1e-4}], weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    best_score, best_epoch, history = float('inf'), None, []
    for epoch in range(1, args.epochs + 1):
        # Frozen BN statistics/dropout avoid treating training noise as crop inconsistency.
        model.eval()
        losses = []
        for a, b, target in training:
            a, b, target = a.to(device), b.to(device), target.to(device).clamp(1., 95.)
            optimizer.zero_grad(set_to_none=True)
            ya, yb = model(a).reshape(-1), model(b).reshape(-1)
            supervised = .5 * (F.smooth_l1_loss(ya, target) + F.smooth_l1_loss(yb, target))
            consistency = F.smooth_l1_loss(ya, yb)
            loss = supervised + args.consistency_weight * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        result = evaluate(model, validation, device)
        score = result['mae'] + .5 * result['mean_crop_range']
        accepted = eligible(result, baseline)
        if accepted and score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), args.output / 'best_candidate.pth')
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **result, 'eligible': accepted})
        save_json(args.output / 'history.json', history)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, args.output / 'last.pt')
        print(json.dumps(history[-1]), flush=True)
    if best_epoch is not None:
        model.load_state_dict(torch.load(args.output / 'best_candidate.pth', map_location='cpu', weights_only=True), strict=True)
        candidate_test = evaluate(model, batches(val.held_out_test), device)
        model.load_state_dict(state, strict=True)
        baseline_test = evaluate(model, batches(val.held_out_test), device)
    else:
        candidate_test = baseline_test = None
    save_json(args.output / 'result.json', {'state': 'complete', 'best_epoch': best_epoch,
              'baseline_test': baseline_test, 'candidate_test': candidate_test,
              'candidate_available': best_epoch is not None,
              'deployment': 'Requires workstation conversion parity and physical-device validation before replacing model'})


if __name__ == '__main__':
    main()
