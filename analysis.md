# 🔍 Analysis: Why Your Colab Results Don't Match the Paper

**Paper**: *"A hybrid CNN-LSTM model for accurate fruit freshness classification using deep learning"*, Ghosh & Singh (2026)
**Paper's best results**: Accuracy 98.9%, Precision 97.5%, Recall 98.1%, F1-Score 97.8%
**Your results**: DenseNet121 Val Accuracy ~73%, Test Accuracy ~52%, multiple models failing to converge

---

## Critical Issues Found

### ❌ Issue 1: WRONG Dataset Split (Most Critical)

**Paper** (Table 3):
| Split | Images | % |
|---|---|---|
| Train | 9,519 | 70% |
| Validation | 1,360 | 10% |
| Test | 2,720 | 20% |
| **Total** | **13,599** | **100%** |

**Your code**: You split *only the apple subset* with `validation_split=0.10`:
```python
# You found:
# Training: 3,632 images (apple only)
# Validation: 403 images  (apple only)
# Test: 996 images         (apple only)
```

The paper uses **all 13,599 images from all 3 fruit types combined** in a single stratified 70/10/20 split, training **one unified binary classifier** (fresh vs rotten) across all 6 classes.

Your notebook trains on **only apples** (~4,000 images) then evaluates on apples-only test set — but in the final evaluation section at the bottom, a different `model` is referenced against a full 6-class test set (which gives 52% accuracy because the model was never trained on bananas/oranges).

> [!CAUTION]
> This single flaw alone explains most of the performance gap. The paper's model sees all fruit types during training.

---

### ❌ Issue 2: WRONG Data Augmentation Parameters

**Paper** (Table 4):
| Augmentation | Parameter |
|---|---|
| Rotation | ±30° |
| Flipping | Horizontal, Vertical |
| Brightness adjustment | ±25% |
| Gaussian noise | Mean=0, Var=0.01 |
| Zoom | 90%–110% |

**Your code**:
```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,        # ✅ correct
    horizontal_flip=True,     # ❌ missing vertical_flip=True
    zoom_range=0.20,          # ❌ should be 0.10 (90-110% = ±10%)
    brightness_range=[0.8, 1.2],  # ✅ correct (±20% close enough to ±25%)
    validation_split=0.10
)
# ❌ Missing: Gaussian noise (paper uses Mean=0, Var=0.01)
```

---

### ❌ Issue 3: WRONG Dataset Split Strategy

The paper uses **stratified sampling** to split the full dataset into train/val/test with proportions 70/10/20. Your code instead:
1. Takes only one fruit's classes (`freshapples`, `rottenapples`)
2. Applies `validation_split=0.10` from Keras (which is NOT stratified)
3. Uses the original Kaggle `test/` folder as the test set

The paper explicitly states: *"The dataset was divided into strictly disjoint training, validation, and test sets using stratified sampling (70%, 10%, 20%), thereby preventing any form of data leakage."*

You should merge the Kaggle train+test folders, then perform a single stratified split using `sklearn.model_selection.train_test_split` across all 13,599 images.

---

### ❌ Issue 4: WRONG CNN-LSTM Architecture

**Paper** (Table 7):
| Component | Details |
|---|---|
| Conv2D Layers | 3–4 layers with increasing filters (16→32→64→128), ReLU |
| Pooling | MaxPooling2D after each Conv2D |
| Batch norm | Applied after Conv2D |
| Dropout | 0.5 |
| Flatten | Converts 2D maps to 1D |
| **LSTM** | **One LSTM layer with 64 units, activation='tanh'** |
| Dense layer | Dense(128, activation='relu') |
| Output | Dense(1, activation='sigmoid') |

**Your code**:
```python
# You use pre-trained backbones (DenseNet121, ResNet50, etc.) + GlobalAveragePooling2D + LSTM(128)
x = TimeDistributed(GlobalAveragePooling2D(), name="td_gap")(x)
x = LSTM(128, return_sequences=False, name="lstm_2")(x)  # 128 units, not 64
```

The paper's **hybrid model** uses a **custom CNN** (not transfer learning) with 3-4 Conv2D layers for the CNN-LSTM hybrid. The paper uses **pre-trained models separately** (Table 6) and the **hybrid CNN-LSTM** (Table 7) as a *different* architecture entirely. You're mixing both.

The paper's CNN-LSTM hybrid uses:
- **Custom CNN** (not VGG16/ResNet etc.) with Conv2D layers as the feature extractor
- **LSTM with 64 units** (not 128)

However, reading more carefully: the paper also tests the CNNs *as feature extractors* with LSTM on top. Table 6 refers to pre-trained models used as standalone classifiers, while Section 3.3 introduces the hybrid which uses a custom CNN backbone.

---

### ❌ Issue 5: Multi-Class Confusion in Binary Classification

The paper's binary classification (fresh vs rotten) consolidates all 6 classes into 2: fresh/rotten.

In your training generator, you correctly do this binary mapping, but your **sequence generator creates a shape mismatch**:

```python
# CRITICAL BUG for InceptionV3:
# ValueError: could not broadcast input array from shape (299,299,3) into shape (224,224,3)
```

Your `make_sequence_generator_binary` always creates `seq_batch` with `IMG_SIZE=(224,224)` but InceptionV3 uses `(299,299)`. This crashes InceptionV3 training entirely.

---

### ❌ Issue 6: Training Data Size is Far Too Small

The paper augments training data from 9,519 → **12,850 images** (Table 5) across all 6 classes. Your apple-only training uses only **3,632 images** before augmentation. This is ~4× fewer samples, which severely limits model generalization.

---

### ❌ Issue 7: Transfer Learning Freezing Strategy

**Paper** (Table 8):
> *"Optimized: Fine-tuned with last 3 convolutional blocks active"*

The paper fine-tunes only the **last 3 convolutional blocks** of the backbone (all others frozen).

**Your code**:
```python
BACKBONE_CONFIG = {
    "DenseNet121": {"freeze_until": 100},   # Freezes first 100 layers
    "VGG16":       {"freeze_until": 15},    # Only freezes 15 layers!
}
```

For VGG16, freezing only 15 layers means most of the network is trainable with limited data, leading to overfitting. The correct approach follows the paper: freeze all layers except the last 3 convolutional blocks.

---

### ❌ Issue 8: Training/Evaluation Mismatch in Final Section

At the end of your notebook (Page 11), you call:
```python
test_seq_gen = make_sequence_generator(test_gen, seq_len=SEQ_LEN)  # undefined function!
y_pred_probs = model.predict(...)
y_pred = np.argmax(y_pred_probs, axis=1)  # multi-class argmax on binary model!
```

Problems:
1. `make_sequence_generator` is **undefined** (you defined `make_sequence_generator_binary`)
2. `np.argmax` is used on a binary sigmoid output — incorrect, should be `(y_prob >= 0.5).astype(int)`
3. The 6-class classification report is applied to a binary model → gives 52% accuracy

---

## Summary Table

| Issue | Paper | Your Code | Impact |
|---|---|---|---|
| Dataset scope | All 13,599 images (all fruits) | ~4,000 images (apples only) | 🔴 Critical |
| Data split | Stratified 70/10/20 across all classes | `validation_split=0.10` on apple subset | 🔴 Critical |
| Augmentation | Vertical + horizontal flip | Horizontal only | 🟡 Medium |
| Zoom | ±10% (90–110%) | ±20% (over-augmentation) | 🟡 Medium |
| Gaussian noise | In augmentation pipeline | Missing | 🟡 Medium |
| LSTM units | 64 | 128 | 🟡 Medium |
| InceptionV3 sequence | Fixed (299×299) | Bug: shape mismatch crash | 🔴 Critical |
| Final eval function | Correct binary eval | Undefined function + argmax bug | 🔴 Critical |
| Freeze strategy | Last 3 blocks unfrozen | Inconsistent per model | 🟠 High |

---

## How to Fix: Step-by-Step

### Step 1: Correct Data Loading (All 13,599 images, stratified split)
```python
import os
from sklearn.model_selection import train_test_split

BASE_DIR = "/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset"
all_images, all_labels = [], []

for split_folder in ['train', 'test']:
    split_path = os.path.join(BASE_DIR, split_folder)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        label = 0 if 'fresh' in class_name.lower() else 1  # binary: 0=fresh, 1=rotten
        for img_name in os.listdir(class_path):
            all_images.append(os.path.join(class_path, img_name))
            all_labels.append(label)

# Stratified 70/10/20 split
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    all_images, all_labels, test_size=0.30, random_state=42, stratify=all_labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp)  # 2/3 of 30% = 20%
# Result: ~70% train, ~10% val, ~20% test
```

### Step 2: Correct Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,        # ±30°
    horizontal_flip=True,     # horizontal flip
    vertical_flip=True,       # ADD: vertical flip
    zoom_range=0.10,          # FIX: 90-110% = ±10%
    brightness_range=[0.75, 1.25],  # ±25%
    # Gaussian noise: add as a layer in the model or use tf.keras.layers.GaussianNoise
)
```

### Step 3: Fix LSTM Units (64, not 128)
```python
x = LSTM(64, return_sequences=False, name="lstm_1")(x)
```

### Step 4: Fix InceptionV3 Sequence Generator
```python
def make_sequence_generator_binary(base_generator, seq_len=SEQ_LEN, 
                                    class_weight=None, img_h=224, img_w=224):
    # Pass img_h, img_w as parameters so InceptionV3 uses 299x299
    for X_batch, y_batch in base_generator:
        B = X_batch.shape[0]
        H, W = X_batch.shape[1], X_batch.shape[2]  # use actual batch dimensions
        seq_batch = np.zeros((B, seq_len, H, W, 3), dtype=np.float32)
        ...
```

### Step 5: Fix Final Evaluation
```python
# Binary evaluation (NOT argmax):
y_prob = model.predict(seq, steps=steps, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
y_true = [0 if 'fresh' in idx_to_class[c].lower() else 1 for c in test_gen.classes]
```

---

## Expected Results After Fixes

Once you fix these issues (especially Issues 1, 2, 4, 5, 8), you should see results approaching the paper's performance:

| Model | Paper Result | Expected After Fix |
|---|---|---|
| InceptionV3 | 89.2% | ~85–90% |
| VGG16 | 85.6% | ~82–87% |
| ResNet50 | 91.1% | ~88–93% |
| DenseNet121 | 91.6% | ~89–93% |
| EfficientNetB0 | 92.4% | ~90–94% |
| **Hybrid CNN-LSTM** | **98.9%** | **~95–99%** |
