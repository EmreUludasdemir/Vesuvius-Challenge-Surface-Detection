# Vesuvius Challenge Surface Detection

Automated surface detection and segmentation for Herculaneum papyrus scrolls from 3D CT scans. This project aims to solve one of the most challenging problems in the Vesuvius Challenge: automatically detecting and mapping papyrus surfaces in damaged, compressed 3D volumes.

## 🎯 Project Overview

Surface detection is the **most critical step** in the virtual unwrapping pipeline. Without accurate surface detection, ink detection cannot proceed because flattened surface volumes cannot be created.

### The Challenge

- **Physical difficulty**: Scrolls carbonized in 79 AD Vesuvius eruption, heavily damaged and compressed
- **Technical difficulty**: Sheet switching (surface tracking jumps between parallel wraps), beam hardening artifacts, cross-fiber edges
- **Scale**: 11-15 meter long spiral-wrapped papyrus layers, impossible to separate manually
- **Cost**: Manual tracking can cost $1-5M per scroll

### This Implementation

State-of-the-art approaches combining classical computer vision and deep learning:
- ✅ **Baseline Sobel detector** - Fast classical CV approach
- ✅ **3D U-Net** - Deep learning segmentation
- ✅ **Z-translation augmentation** - Critical for performance
- ✅ **Combined BCE + Dice Loss** - Optimal loss function
- ✅ **Depth-invariant architecture** - Based on Kaggle winners

## 📚 Documentation

- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Comprehensive research report covering:
  - State-of-the-art methods (ThaumatoAnakalyptor, Surface Tracer, Spiral Fitting)
  - Kaggle winner strategies and model architectures
  - Critical success factors and augmentation techniques
  - Training strategies and hyperparameters
  - Known pitfalls and how to avoid them

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/EmreUludasdemir/Vesuvius-Challenge-Surface-Detection.git
cd Vesuvius-Challenge-Surface-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

Download fragment data from:
- [Vesuvius Challenge Data Server](https://dl.ash2txt.org/)
- [Kaggle Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)

Place data in `data/raw/` directory:
```
data/raw/
├── fragment_1/
│   ├── surface_volume/
│   │   ├── 00.tif
│   │   ├── 01.tif
│   │   └── ...64.tif
│   ├── mask.png
│   └── inklabels.png
```

### Explore Data

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Train Baseline Model

```bash
# Test Sobel baseline
python src/models/sobel_baseline.py

# Train 3D U-Net (coming soon)
# python train.py --config configs/default_config.yaml
```

## 📁 Project Structure

```
Vesuvius-Challenge-Surface-Detection/
├── configs/                    # Configuration files
│   ├── default_config.yaml
│   ├── low_memory_config.yaml
│   └── high_performance_config.yaml
├── data/                       # Data directory
│   ├── raw/                    # Raw CT volumes
│   ├── processed/              # Processed data
│   └── outputs/                # Model outputs
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── src/                        # Source code
│   ├── data/                   # Data processing
│   │   ├── preprocessing.py    # Volume loading, patch extraction
│   │   └── augmentations.py    # Z-translation, 2D augmentations
│   ├── models/                 # Model architectures
│   │   ├── sobel_baseline.py   # Classical CV baseline
│   │   └── unet3d.py           # 3D U-Net models
│   ├── training/               # Training utilities
│   │   ├── losses.py           # Loss functions
│   │   └── trainer.py          # Training loop
│   └── utils/                  # Utilities
├── TECHNICAL_GUIDE.md          # Comprehensive technical documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧠 Model Architectures

### 1. Baseline: 3D Sobel Detector

Classical computer vision approach based on ThaumatoAnakalyptor:
- 3D Sobel gradient computation
- Gradient magnitude thresholding
- Optional umbilicus direction filtering
- Morphological operations
- Point cloud extraction

**Pros**: Fast, no training required
**Cons**: Struggles with sheet switching, beam hardening artifacts

### 2. 3D U-Net (Depth-Invariant)

Deep learning approach based on Kaggle winner strategies:
- Processes 65 depth slices as input channels
- Encoder-decoder architecture with skip connections
- Outputs 2D surface probability map
- Depth-invariant design (recommended)

**Pros**: Better accuracy, learns from data
**Cons**: Requires training data and GPU

### 3. Two-Stage Architecture (Future)

State-of-the-art approach from 1st place Kaggle solution:
- Stage 1: 3D feature extraction (3D U-Net/UNETR)
- Stage 2: 2D segmentation (SegFormer)
- Ensemble of 5-9 models

**Expected performance**: ~0.68-0.74 CV score

## 🔑 Critical Success Factors

### 1. Z-Translation Augmentation ⭐ MOST IMPORTANT

Single biggest performance boost. Shifts slices along depth axis for translation invariance.

```python
from src.data.augmentations import ZTranslationAugment

z_aug = ZTranslationAugment(max_shift=5, p=0.5)
augmented_volume = z_aug(volume)
```

### 2. Small Context Windows

Counterintuitive but **64×64 > 256×256** in reliability. Large windows cause:
- Fixed brush width bias
- Letter completion/filling artifacts
- Less reliable stroke interpretation

### 3. Combined BCE + Dice Loss

```python
from src.training.losses import CombinedLoss

loss_fn = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5,
    label_smoothing=0.1
)
```

### 4. Heavy Augmentations

- Rotation and flip (mandatory)
- CoarseDropout
- Elastic transforms
- Per-channel transformations

### 5. Label Smoothing

Essential for preventing overfitting. Use `label_smoothing=0.1`.

## ⚙️ Configuration

Three pre-configured setups:

### Default (24GB GPU)
```bash
python train.py --config configs/default_config.yaml
```
- Batch size: 4
- Base features: 32
- Patch size: 256

### Low Memory (16GB GPU)
```bash
python train.py --config configs/low_memory_config.yaml
```
- Batch size: 2
- Base features: 16
- Patch size: 128
- Gradient accumulation: 2

### High Performance (40GB+ GPU)
```bash
python train.py --config configs/high_performance_config.yaml
```
- Batch size: 16
- Base features: 64
- Patch size: 512

## 📊 Expected Performance

Based on similar Kaggle Ink Detection challenge:

| Approach | Expected CV Score | Training Time |
|----------|------------------|---------------|
| Sobel Baseline | N/A (no training) | Instant |
| Single 3D U-Net | 0.65-0.70 | ~10 hours |
| Two-Stage (U-Net + SegFormer) | 0.68-0.72 | ~20 hours |
| Full Ensemble (5-9 models) | 0.68-0.74 | ~100 hours |

## 🎓 Key Learnings from Research

1. **Z-translation is critical** - Single most important augmentation
2. **Smaller is better** - 64×64 windows more reliable than 256×256
3. **Ensemble diversity matters** - Mix different architectures
4. **Label smoothing essential** - Prevents overfitting
5. **Domain shift is real** - Fragment models often fail on scrolls

## 🚧 Known Issues & Pitfalls

### Training Issues
- ⚠️ Overfitting on small datasets → Use label smoothing + heavy augmentations
- ⚠️ Large window size → Causes letter completion artifacts
- ⚠️ No Z-translation → Poor depth invariance

### Domain-Specific Challenges
- ⚠️ Sheet switching → Requires post-processing continuity checks
- ⚠️ Beam hardening artifacts → Careful gradient thresholding
- ⚠️ Cross-fiber edges → False positives
- ⚠️ Fragment → Scroll gap → Needs domain adaptation

## 🔗 Resources

### Official Vesuvius Challenge
- [Vesuvius Challenge Website](https://scrollprize.org/)
- [Data Server](https://dl.ash2txt.org/)
- [Discord Community](https://discord.gg/6FgWYNjb4N)

### Referenced Methods
- [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor)
- [Volume Cartographer](https://github.com/educelab/volume-cartographer)

### Kaggle Solutions
- [1st Place Ink Detection](https://github.com/ainatersol/Vesuvius-InkDetection)
- [6th Place Solution](https://github.com/chumajin/kaggle-VCID)
- [Grand Prize Winner](https://github.com/younader/Vesuvius-Grandprize-Winner)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{vesuvius_surface_detection,
  title = {Vesuvius Challenge Surface Detection},
  author = {Emre Uludasdemir},
  year = {2024},
  url = {https://github.com/EmreUludasdemir/Vesuvius-Challenge-Surface-Detection}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Vesuvius Challenge organizers and prize sponsors
- ThaumatoAnakalyptor team (Julian Schilliger)
- Surface Tracer team (Dr. Hendrik Schilling & Sean Johnson)
- Spiral Fitting team (Prof. Paul Henderson)
- Kaggle Ink Detection competition winners
- Grand Prize winners (Youssef Nader et al.)

## 📧 Contact

For questions or discussions, please open an issue or reach out on the Vesuvius Challenge Discord.

---

**Note**: This is an active research project. The 2024 First Automated Segmentation Prize was not won, indicating how challenging this problem is. Contributions and improvements are highly encouraged!
