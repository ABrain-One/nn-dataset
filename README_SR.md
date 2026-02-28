# Quick Reference Guide - MAI 2025 SR Implementation

## 📁 Important Files

### Documentation
- `THESIS_IMPLEMENTATION.md` - Complete technical documentation
- `EMAIL_TO_PROFESSOR.md` - Summary email for professor
- `README.md` - This quick reference

### Models
- `ab/nn/nn/RLFN.py` - RLFN model (MAI 2022)
- `ab/nn/nn/SPAN.py` - SPAN model (NTIRE 2024)

### Data & Metrics
- `ab/nn/loader/div2k.py` - DIV2K dataset loader
- `ab/nn/metric/psnr.py` - Y-channel PSNR
- `ab/nn/metric/ssim.py` - Y-channel SSIM

### Results
- `ab/nn/stat/train/img-sr_div2k_psnr_RLFN/` - RLFN+PSNR results
- `ab/nn/stat/train/img-sr_div2k_psnr_SPAN/` - SPAN+PSNR results

---

## 🚀 Quick Commands

### Train RLFN (10 epochs)
```bash
PYTHONPATH=. python ab/nn/train.py --config "img-sr_div2k_psnr_RLFN" --epochs 10 --trials 1
```

### Train SPAN (10 epochs)
```bash
PYTHONPATH=. python ab/nn/train.py --config "img-sr_div2k_psnr_SPAN" --epochs 10 --trials 1
```

### Run Benchmarks
```bash
PYTHONPATH=. python ab/nn/util/Benchmark.py
```

### Demo Inference
```bash
PYTHONPATH=. python demo/sr_demo.py
```

---

## 📊 Benchmark Results

| Model | Params | FLOPs | Latency | PSNR* |
|-------|--------|-------|---------|-------|
| RLFN  | 450K   | 2.08G | 7.51ms  | 9.55  |
| SPAN  | 605K   | 2.87G | 9.04ms  | 0.11  |

*With synthetic data. Real DIV2K: 25-35 dB expected

---

## ✅ Verification Checklist

- [x] Models in `ab/nn/nn/`
- [x] Loader in `ab/nn/loader/`
- [x] Metrics in `ab/nn/metric/`
- [x] Results in `ab/nn/stat/train/`
- [x] Empty `__init__.py`
- [x] Training works
- [x] Benchmarks auto-collected

---

## 🤔 Open Questions

1. **Checkpoints**: Save .pth files? Where?
2. **Real Data**: Download DIV2K dataset?
3. **More Models**: Add SAFMN, CARN-M?
4. **Evaluation**: Visual comparisons? Other datasets?

---

## 📞 Contact

Mohsin Ikram - mohsin271998@gmail.com
