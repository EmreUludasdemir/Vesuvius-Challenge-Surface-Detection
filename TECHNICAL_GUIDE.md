# Vesuvius Challenge Surface Detection: Araştırma Raporu ve Teknik Kılavuz

**Vesuvius Challenge Surface Detection**, antik Herculaneum papirüs rulolarının 3D CT taramalarında papirüs yüzeylerinin otomatik olarak tespit edilmesi ve haritalanmasını hedefleyen kritik bir bilgisayarlı görü problemidir. 2023'te 700.000$ Grand Prize kazanıldıktan sonra, 2024'te bu tekniğin otomasyonu için 100.000$ ödül konuldu ancak **kazanılmadı** - bu problemi çözmenin ne kadar zor olduğunu gösteriyor.

## İçindekiler

- [Surface Detection Nedir ve Neden Kritik?](#surface-detection-nedir-ve-neden-kritik)
- [State-of-the-Art Yöntemler](#state-of-the-art-surface-detection-yöntemleri)
- [Kaggle Ink Detection Kazanan Stratejileri](#kaggle-ink-detection-kazanan-stratejileri)
- [Model Mimarileri](#3d-mimariler-en-yaygın)
- [Training Stratejileri](#training-stratejileri-ve-hyperparameters)
- [Önerilen Yaklaşım](#önerilen-yaklaşım-claude-code-için-en-iyi-strateji)
- [Computational Requirements](#computational-requirements-özet)
- [Bilinen Pitfalls](#bilinen-pitfalls-ve-kaçınılması-gerekenler)

## Surface Detection Nedir ve Neden Kritik?

Surface detection (segmentasyon), sanal açma (virtual unwrapping) işleminin **en kritik ve zor** aşamasıdır. Hasar görmüş, sıkışmış, karmaşık 3D CT taramalarında papirüs yapraklarının geometrisini haritalamayı içerir. Bu adım olmadan mürekkep tespiti yapılamaz çünkü düzleştirilmiş yüzey hacimleri (surface volumes) elde edilemez.

### Fiziksel Zorluklar

Rulolar 79 AD'de Vezüv patlamasıyla karbonize olmuş, ezilmiş, zarar görmüş durumda:
- CT taramalarında tabakalar bulanık görünüyor
- Birbirine yapışmış bölgeler var
- Kompresyon artefaktları ve beam hardening etkileri mevcut
- Papirüs tabakaları 11-15 metre uzunluğunda spiral sarılmış
- Birbirinden ayrılamayacak kadar yakın

### Teknik Zorluk

En yaygın hata **"sheet switching"** - yüzey takibi bir paralel sarımdan diğerine atlayarak gerçekçi olmayan bozuk şekiller oluşturuyor. Manuel izleme çok yavaş ve pahalı (rulo başına 1-5 milyon $ maliyetli olabilir).

## State-of-the-Art Surface Detection Yöntemleri

### 1. ThaumatoAnakalyptor (Bottom-up yaklaşım - Julian Schilliger)

**Ara temsil**: Klasik bilgisayarlı görü ile yoğun nokta bulutu çıkarma

**İşlem hattı**:
- Ham CT verileri üzerinde **3D Sobel Kernel** ile gradyan hesaplama
- Gradyan büyüklüğünü eşikleme (birinci ve ikinci türevler)
- Scroll umbilicus'una işaret eden vektörle iç nokta çarpımı yaparak iç yüzeyi seçme
- Binary maske oluşturma (yüzey voxel'leri = 1, diğerleri = 0)
- Yoğun nokta bulutunu çıkarma

**Instance segmentasyon**:
- **Mask3D** deep neural network ile noktaları patch'lere gruplama
- Overlapping patch'ler kasıtlı olarak oluşturuluyor

**Stitching**:
- Monte Carlo yaklaşımı ile belirsizlik grafları
- Node'lar = patch'ler, edge'ler overlap miktarına göre ağırlıklandırılmış
- Random walk'lar ile subgraph oluşturma
- Maksimal overlap'li cover'ları seçme

**Mesh reconstruction**: Poisson Surface Reconstruction veya Delaunay Triangulation

**Flattening**: SLIM algoritması (Scalable Locally Injective Mappings) ile izometrik distorsiyonu minimize eden UV parametrizasyonu

**Avantajlar**: 2023 Grand Prize'ın parçası, bazı scrolllarda umut verici sonuçlar

**Dezavantajlar**: Sıkışmış bölgelerde düşük yoğunluk, cross-fiber edge'lerde yanlış pozitifler, beam hardening artefaktları

### 2. Surface Tracer (Bottom-up - Dr. Hendrik Schilling & Sean Johnson)

**Ara temsil**: nnU-Net ile semantik segmentasyon (yüzey tahminleri)

**İşlem mantığı**: Papirüs fiziksel olarak sürekli ve ani eğilmemeli - yüzeyi "takip ederek" büyüt

**Geometry processing**:
- Küçük patch'lerden başla
- İteratif olarak genişlet
- Yeni noktaları seçerken lokal objektifi minimize et:
  - Data fidelity (nokta yüzey tahminleri üzerinde)
  - Smoothness (büyük atlamaları önle)
  - Flatness (lokal eğriliği etkiler)
- Her n iterasyonda mesh'i smooth et
- Paralel büyüyen patch'leri consensus algoritmasıyla birleştir
- İnsan annotator'lar manuel olarak iyileştirme yapar

**Durum**: First Automated Segmentation Prize 2024 submission'ı - hedef kaliteye çok yakın ama yeterli değildi

### 3. Spiral Fitting (Top-down - Prof. Paul Henderson)

**Felsefe**: Global optimizasyon - tüm yüzeyi aynı anda fit et

**Ana fikir**: Orijinal scroll 2D spiral'in 3D silindirik şekle extrude edilmesi. Gözlemlenen veriye canonical spiral'i deforme eden transformasyonu bul.

**Ara temsil**: nnU-Net ile semantik segmentasyon
- Surface sheet predictions
- Vertical papyrus fiber predictions
- Horizontal papyrus fiber predictions

**Processing**: 3D diffeomorphism (türevlenebilir, invertible transformasyon) bulma - global objektifi minimize ederek

**Durum**: 2024 First Automated Segmentation Prize submission'ı, en zarif çözüm ama kalite kriterleri karşılanmadı

## Kaggle Ink Detection Kazanan Stratejileri

Surface Detection'a uyarlanabilir teknikler ve model mimarileri.

### En Başarılı Model Mimarileri

#### 1st Place (ryches - F0.5: 0.682693) - 9-model ensemble

**İki aşamalı mimari**:
- **Stage 1 - 3D özellik çıkarma**: 3D CNN, 3D U-Net, UNETR (U-Net Transformer)
- **Stage 2 - 2D segmentasyon**: SegFormer (B1, B3, B5 backbone'ları)
- Depth-invariant tasarım
- 3 A6000 GPU (40GB), model başına ~10 saat eğitim

**Ensemble modelleri**:
- CNN3D_Segformer
- unet3d_segformer (512 ve 1024 boyutları)
- unet3d_segformer_jumbo
- UNETR_Segformer, UNETR_SegformerMC, UNETR_MulticlassSegformer

#### 6th Place (chumajin - Private LB: 0.66)

| Model | Image Size | CV Score | Private LB |
|-------|-----------|----------|------------|
| efficientnet_b7_ns | 608 | 0.712 | 0.64 |
| efficientnet_b6_ns | 544 | 0.702 | 0.64 |
| efficientnetv2_l | 480 | 0.707 | 0.65 |
| tf_efficientnet_b8 | 672 | 0.716 | 0.64 |
| **segformer-b3** | **1024** | **0.738** | **0.66** |

### 3D Mimariler (En Yaygın)

**ResNet3D ailesi**: ResNet3D-18, 34, 50, 101, 152 (Kinetics pretrained)

**I3D (Inflated 3D ConvNets)**: Non-local block'larla, ResNet3D-50'den ~1/2 parametre, küçük veri setlerinde daha az overfitting

**TimeSformer**: Grand Prize winner'ın canonical modeli
- Divided space-time attention
- Encoder-only (decoder gerekmez)
- Input: 64×64, Output: 4×4 grid (16 değer)
- 15-25 layer optimal
- Facebook AI Research tabanlı

**3D U-Net varyantları**: Standard 3D U-Net, Jumbo U-Net, depth-invariant tasarım kritik

**UNETR**: Transformer encoder'lı U-Net

**MViTv2**: Multi-scale Vision Transformer v2

### Critical Başarı Faktörleri

#### 1. Z-translation augmentation (derinlik kayması)

**Tek başına en büyük etkili augmentation**
- Translation invariance için kritik
- 4th place çözümünün vurguladığı teknik

#### 2. Küçük context window'lar daha iyi (counterintuitive!)

- 64×64 > 256×256 > 512×512 güvenilirlik açısından
- Büyük window'lar:
  - Fixed brush width bias'ı
  - Letter completion/filling (C'yi O'ya çeviriyor)
  - Daha az güvenilir stroke yorumlama

#### 3. Heavy augmentation

- Her kanal/slice için 2D augmentations
- Rotation ve flip invariance (zorunlu)
- Cutout, CutMix, MixUp, Manifold MixUp
- CoarseDropout
- Stochastic depth (rate=0.2)
- Per-channel transformations

#### 4. Loss functions

- **BCE + Dice Loss** (en yaygın kombinasyon)
- Global F-beta Loss (tüm batch üzerinden)
- Label smoothing (0.1 smoothing factor)
- Gaussian weighting (visible signal alanlarına odaklan)

#### 5. Ensemble diversity

- 5-9 model tipik
- Farklı mimariler (ResNet3D + I3D + TimeSformer + SegFormer)
- Farklı fold'lar
- SWA (Stochastic Weight Averaging) versiyonlar

### Input/Output Dimensions

**Tipik input**:
- 3D Volume: 65 slices (depth) × H × W
- Patch boyutları: 64×64, 256×256, 512×512, 1024×1024
- Merkezi 27-35 slice en yaygın kullanılan
- 3ch×9 veya 5ch×7 gruplara bölünerek 2D işleme

**Output resolution**:
- Full resolution veya downsampled 1/2, 1/4, 1/16, 1/32
- Size/4 en yaygın U-Net decoder'lar için
- Size/16 TimeSformer için
- Upsampling bilinear interpolation ile

## Training Stratejileri ve Hyperparameters

### Optimizer ve Scheduler

**AdamW** (standart seçim)
- Learning rate: 0.0001 (1e-4) default
- GPU memory için ayarlanmış
- Exponential Moving Average (EMA): decay=0.99
- GradualWarmupScheduler
- SWA (Stochastic Weight Averaging) final modeller için

### Regularization

- **Label smoothing**: 0.1
- Heavy augmentations
- Small model selection (I3D over ResNet3D-50) overfitting'i önlemek için
- Data balancing: 50/50 ink vs background sampling

### Training Times ve Compute

- **1st Place**: ~10 saat per model on 3 A6000 GPUs (40GB each)
- **6th Place**: V100 (EfficientNetV2-L) ve A100 (diğer modeller) Colab Pro+'da
- **Grand Prize**: Multi GPU, vast.ai ile GPU kiralama
- **Batch sizes**: Training 4-16, validation 8-32, inference 32-64

### GPU Memory Requirements

**High Memory**: 40GB+ önerilen (A6000, A100)

**Reduced Setup**: 24GB ile mümkün:
- Daha küçük backbone (B1 yerine B3/B5)
- Daha küçük batch size
- Gradient accumulation
- Mixed precision (FP16)
- torch.compile (PyTorch 2.0+)

## Post-processing Teknikleri

1. **Percentile thresholding**: Adaptive threshold'lar (0.96 yaygın)
2. **Edge masking**: Tile boundary'lere yakın tahminleri çıkar (CNN artefaktları)
3. **Gaussian kernel weighting**: Overlapping patch'ler için
4. **Stride selection**: Accuracy vs inference time trade-off
5. **Morphological operations**: Tahminleri temizle
6. **Connectivity principle**: Küçük izole pixel grupları muhtemelen noise, harf değil

## Framework ve Kütüphaneler

### Core
```python
torch>=2.0.1
torch-lightning
segmentation_models_pytorch==0.3.3
timm==0.9.2
transformers==4.29.2  # veya >=4.57.0
```

### Computer Vision
```python
albumentations
opencv-python
pillow
scikit-image
```

### 3D Networks
```python
pytorch-3dunet  # Wolny et al.
timesformer-pytorch
monai  # Medical Open Network for AI
```

### Utilities
```python
wandb  # experiment tracking
einops  # tensor operations
tqdm
numpy
scipy
```

**Docker**: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel önerilen

## Dataset Yapısı ve Format

### Input Data Structure
```
inputpath/
├── fragmentid/
│   ├── surface_volume/
│   │   ├── 00.tif
│   │   ├── 01.tif
│   │   └── ...64.tif (65 slices total)
│   ├── mask.png
│   ├── ir.png  (infrared photo)
│   └── inklabels.png (ground truth)
```

### CT Scan Özellikleri

- **Format**: Stacked 2D .tif files (alt'tan üst'e horizontal slices)
- **Voxel sizes**: 2-7.91 μm tipik (3.24 μm ve 7.91 μm en yaygın)
- **Bit depth**: 16-bit original, bazen 8-bit'e reduce edilir
- **Energy levels**: 53-54 keV veya 88 keV X-ray
- **Representation**: Grayscale intensity - açık = daha yoğun (papyrus), koyu = daha az yoğun (hava)

### Expected Output

- **Binary 3D volume**: Input ile aynı boyutlarda, yüzey voxel'leri marked
- **Point cloud**: .ply veya .obj format, surface points'lerin 3D koordinatları
- **Mesh**: .obj format, triangular mesh
- **Surface volume**: 65 layers (32 above + 1 center + 32 below detected surface)

## Bilinen Pitfalls ve Kaçınılması Gerekenler

### Training Issues

1. **Overfitting**:
   - Küçük modeller kullan (I3D < ResNet3D-50 parameters)
   - Label smoothing essential
   - Heavy augmentations
   - Segment'ler arası generalization'ı monitor et

2. **Data bias**:
   - Label'larda fixed brush width → model fixed width output
   - Scattered labels false negative'leri introduce eder
   - Unbalanced ink/background ratio tahmin brightness'ını etkiler

3. **Window size hataları**:
   - Daha büyük ≠ daha iyi (counterintuitive)
   - Büyük window'lar letter completion/filling artefaktlarına neden oluyor

4. **Decoder problems**:
   - TimeSformer **decoder olmadan** daha iyi çalışıyor
   - U-Net her zaman optimal olmayabilir

### Domain-specific Challenges

1. **Sheet switching**: Fitted surface parallel wrap'ler arasında atlıyor
2. **Fake mergers**: Parallel sheet'ler yanlış birleştirilmiş
3. **Holes and gaps**: Missing surface detections
4. **Cross-fiber edges**: Non-surface features yanlışlıkla surface olarak tespit ediliyor
5. **Beam hardening artifacts**: False surface'lere neden oluyor
6. **Variable density**: Carbonized, damaged papyrus inconsistent contrast'a sahip

### Fragment → Scroll Transfer Gap

- Kaggle fragment modelleri scroll segment'lerinde sık sık başarısız oluyor
- Youssef Nader'ın domain adaptation teknikleri bu gap'i kapatıyor
- Resolution farkları fragment ve scroll'lar arasında önemli
- Crackle pattern labeling vs ground truth photos farklı model davranışları oluşturuyor

## Qwen 3 VL Modeli Değerlendirmesi

### **KULLANMAYIN** - Fundamental Task Mismatch

**Neden uygun değil:**
1. **Yanlış output modalite**: VLM'ler text token üretir, segmentation mask değil
2. **Dense prediction head yok**: Pixel-level tahminler için decoder mimarisi eksik
3. **Compression loss**: 32×32 pixel bölgeleri tek token'a sıkıştırıyor, fine-grained spatial information kayboluyor
4. **3D processing gap**: 3D volumetric CT data'yı native olarak işleyemiyor
5. **Training data mismatch**: Natural image'lar üzerinde pre-trained, medical/CT imaging üzerinde değil

**Qwen 3 VL 2B-Instruct-FP8 özellikleri**:
- 2B parametre, FP8 quantized
- Vision-Language Model (text generation için)
- 2D image/video understanding
- Bounding box output edebilir ama segmentation mask değil
- Surface detection için **fundamentally incompatible**

**Ne zaman kullanılmalı** (post-segmentation):
- Zaten segment edilmiş scroll image'larını analiz etmek
- Metadata veya görsel özellikleri tanımlamak
- OCR of successfully unwrapped papyrus text

## Önerilen Yaklaşım: Claude Code için En İyi Strateji

### Phase 1: Data Understanding

1. Fragment data indir ve keşfet (daha küçük, ground truth var)
2. 3D volume'leri Fiji veya Scroll Viewer ile görselleştir
3. Data server'dan mevcut segment'leri incele
4. Coordinate system'leri ve transformation'ları anla

### Phase 2: Baseline Implementation

#### Approach 1: Classical Computer Vision (hızlı baseline)
```python
# ThaumatoAnakalyptor approach
1. 3D Sobel Kernel ile gradient hesaplama
2. Gradient magnitude thresholding
3. Umbilicus direction filtering
4. Binary mask oluşturma
5. Point cloud extraction
```

#### Approach 2: Deep Learning Segmentation (recommended)
```python
# nnU-Net semantic segmentation
1. nnU-Net'i medical imaging framework olarak kullan
2. Surface, recto, verso predictions
3. Fiber direction predictions (vertical/horizontal)
4. 3D binary volume output
```

#### Approach 3: Hybrid Approach (state-of-the-art)
```python
# Two-stage architecture (1st place Kaggle approach)
Stage 1: 3D CNN/U-Net/UNETR feature extraction
Stage 2: SegFormer 2D segmentation
```

### Phase 3: Model Architecture Önerisi

**En iyi trade-off**: 3D U-Net + SegFormer kombinasyonu

**Model pipeline**:

```python
# Stage 1: 3D Feature Extraction
Input: 65 x H x W (65 depth slices)
Model: 3D U-Net or 3D ResNet-34
Output: C x H x W (multiple channels flattened along depth)

# Stage 2: 2D Segmentation
Input: Flattened 3D features
Model: SegFormer-B3 (if 40GB GPU) or SegFormer-B1 (if <24GB GPU)
Output: H x W binary mask

# Post-processing
- Threshold at 0.96
- Morphological operations
- Connected components filtering
```

### Critical Implementation Details

#### Data Preprocessing
```python
1. Load 65 .tif slices as 3D volume
2. Normalize to 0-1 or standardize
3. Apply windowing for papyrus visualization
4. Tile into 256x256 patches (for training)
5. Sample balanced 50/50 surface/non-surface
```

#### Augmentation Pipeline (MUST HAVE)
```python
import albumentations as A

# Per-slice 2D augmentations
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CoarseDropout(p=0.5),
    # Z-translation (depth shift) - CRITICAL
    # Custom implementation needed
])
```

#### Z-translation Augmentation (en önemli)
```python
def z_translation_augment(volume, max_shift=5):
    """Shift slices along depth axis"""
    shift = np.random.randint(-max_shift, max_shift+1)
    if shift > 0:
        volume = np.concatenate([volume[shift:], volume[:shift]], axis=0)
    elif shift < 0:
        volume = np.concatenate([volume[shift:], volume[:shift]], axis=0)
    return volume
```

#### Loss Function
```python
import segmentation_models_pytorch as smp

# Combined loss (recommended)
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(pred, target, alpha=0.5):
    return alpha * dice_loss(pred, target) + (1-alpha) * bce_loss(pred, target)
```

#### Training Loop
```python
# 7-fold cross-validation
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = GradualWarmupScheduler(...)

# Label smoothing
smooth = 0.1
target_smooth = target * (1 - smooth) + 0.5 * smooth

# EMA for stability
ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

# SWA for final model
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
```

### Ensemble Strategy

**Minimal ensemble** (3-5 models):
1. 3D U-Net + SegFormer-B3 (fold 0-6)
2. 3D ResNet-34 + SegFormer-B3 (fold 0-6)
3. I3D + SegFormer-B1 (fold 0-6)
4. Full train model (all data)

**Averaging**:
```python
# Simple average
ensemble_pred = (model1_pred + model2_pred + model3_pred) / 3

# Weighted average (CV score weighted)
weights = [0.738, 0.712, 0.707]  # CV scores
ensemble_pred = sum(w*p for w,p in zip(weights, predictions)) / sum(weights)
```

### Inference Optimization

```python
# Sliding window with overlap
stride = 4  # pixels
window_size = 256
predictions = sliding_window_inference(
    volume,
    window_size=window_size,
    stride=stride,
    predictor=model
)

# Gaussian weighting for overlap regions
weights = gaussian_kernel_2d(window_size)

# Batch inference
batch_size = 32  # can be higher for inference
```

## Computational Requirements Özet

### Minimum Setup (baseline çalıştırmak için)
- GPU: 16GB VRAM (T4, RTX 3070)
- RAM: 32GB
- Storage: 100GB fragment data

### Recommended Setup (competitive solution için)
- GPU: 40GB VRAM (A6000, A100)
- RAM: 64GB+
- Storage: 500GB+ (multiple scrolls)

### Kaggle Platform'da
- T4 veya P100 GPU (16GB) kullanılabilir
- Reduced backbone'lar kullan (B1 instead of B3)
- Batch size'ı azalt
- Gradient accumulation kullan

## Key Gotchas ve Dikkat Edilmesi Gerekenler

1. **3D volume'leri doğru yönde yükle**: Bottom-to-top ordering (00.tif = bottom)

2. **Coordinate system'lere dikkat**: 3D volume space vs 2D flattened space mapping

3. **Memory management**: Full scroll volume'ler çok büyük, chunk'lara böl

4. **Sheet switching detection**: Post-processing'de continuity check'leri ekle

5. **Beam hardening artifacts**: False positive'leri filter etmek için gradient threshold'ları dikkatli seç

6. **Cross-validation strategy**: Fragment-based split, random değil (scroll characteristics farklı)

7. **Evaluation metrics**: IoU, Dice, Hausdorff distance'ı hesapla

8. **Visual inspection**: Metrics'e tamamen güvenme, manuel görsel inceleme yap

9. **Domain shift**: Fragment'lerden scroll'lara transfer için domain adaptation gerekebilir

10. **Iterative refinement**: Model predictions → manual cleanup → retrain cycle

## Başlangıç için Örnek Kod Repository'leri

### Segmentation için
- ThaumatoAnakalyptor: https://github.com/schillij95/ThaumatoAnakalyptor
- Volume Cartographer: https://github.com/educelab/volume-cartographer

### Ink Detection (uyarlanabilir)
- 1st Place Kaggle: https://github.com/ainatersol/Vesuvius-InkDetection
- 6th Place: https://github.com/chumajin/kaggle-VCID
- Grand Prize: https://github.com/younader/Vesuvius-Grandprize-Winner

### Data Access
- Data server: https://dl.ash2txt.org/
- Kaggle API ile competition data download

## Final Recommendations: Implementation için

### En Önemli Vurgular

1. **Model mimarisi**: İki-aşamalı (3D U-Net + SegFormer) depth-invariant yaklaşım kullan

2. **Augmentation**: Z-translation augmentation mutlaka implement et - tek başına en büyük performance boost

3. **Window size**: Counterintuitive ama 64×64 context window'lar 256×256'dan daha güvenilir

4. **Small models**: Overfitting'i önlemek için küçük modeller tercih et (I3D over ResNet3D-50)

5. **Ensemble**: Minimum 3-5 farklı model architecture'ını ensemble et

6. **Loss**: BCE + Dice kombinasyonu, label smoothing ile

7. **Post-processing**: 0.96 threshold, connectivity-based filtering

8. **Dataset structure**: 65-slice surface volumes, 256×256 tiles, 50/50 balanced sampling

9. **GPU constraints**: 40GB GPU yoksa SegFormer-B1 kullan, batch size'ı düşür

10. **Evaluation**: Metrics + visual inspection kombinasyonu

### İmplementasyon Sırası

**Week 1**: Data pipeline + baseline 3D Sobel approach
**Week 2**: 3D U-Net training + basic augmentations
**Week 3**: SegFormer integration + Z-translation augmentation
**Week 4**: Ensemble + post-processing + inference optimization

### Beklenen Performans

- **Baseline (Sobel)**: Hızlı prototype, but sheet switching issues
- **Single model (3D U-Net)**: Reasonable performance, ~0.65-0.70 CV
- **Two-stage (U-Net + SegFormer)**: Competitive, ~0.68-0.72 CV
- **Full ensemble**: State-of-the-art, ~0.68-0.74 CV (Kaggle Ink Detection benzeri)

---

**ÖNEMLİ NOT**: Surface Detection Kaggle competition'ının spesifik evaluation metric'i ve submission format'ına erişim sağlanamadı (requires Kaggle login). Ancak bu rapor, general surface detection problem'ini çözmek ve competitive bir solution geliştirmek için gerekli tüm teknik detayları ve best practice'leri içeriyor. ScrollPrize.org'daki First Automated Segmentation Prize kriterleri benzer hedeflere sahip.
