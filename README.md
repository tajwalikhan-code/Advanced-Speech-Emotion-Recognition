<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=🎙️%20Speech%20Emotion%20Recognition&fontSize=40&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Advanced%20Deep%20Learning%20System%20%7C%20CodeAlpha%20ML%20Internship&descAlignY=55&descSize=18" width="100%"/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)

<!-- Badges Row 2 -->
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-8A2BE2?style=for-the-badge&logo=data&logoColor=white)](https://zenodo.org/record/1188976)
[![License](https://img.shields.io/badge/License-Educational-27AE60?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete%20✅-2ECC71?style=for-the-badge)](.)

<!-- Badges Row 3 -->
![Emotions](https://img.shields.io/badge/Emotions-8%20Classes-E74C3C?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-SE--CNN%20%2B%20MHA%20%2B%20BiLSTM-9B59B6?style=flat-square)
![Loss](https://img.shields.io/badge/Loss-Focal%20Loss-3498DB?style=flat-square)
![Augmentation](https://img.shields.io/badge/Augmentation-SpecAugment%20%2B%204%20Methods-F39C12?style=flat-square)

<br/>

> **"Teaching machines to understand the emotion behind the words — not just the words themselves."**

<br/>

**Developer:** [Taj Wali Khan](https://linkedin.com/in/tajwalikhan) &nbsp;|&nbsp;
**Internship:** CodeAlpha Machine Learning — Task 2 &nbsp;|&nbsp;
**Platform:** Google Colab (T4 GPU)

</div>

---

## 📌 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [🌟 Project Overview](#-project-overview)
- [🔬 What Makes This Advanced](#-what-makes-this-advanced)
- [🎭 The 8 Emotion Classes](#-the-8-emotion-classes)
- [📊 Dataset — RAVDESS](#-dataset--ravdess)
- [🧠 Model Architecture](#-model-architecture)
- [🎛️ 3-Channel Input Design](#-3-channel-input-design)
- [⚡ Data Augmentation Pipeline](#-data-augmentation-pipeline)
- [🔥 Focal Loss — Why Not CrossEntropy?](#-focal-loss--why-not-crossentropy)
- [📈 Training Strategy](#-training-strategy)
- [📊 Results & Evaluation](#-results--evaluation)
- [🖥️ Gradio Web App — 6 Tabs](#-gradio-web-app--6-tabs)
- [🚀 Quick Start — Run on Colab](#-quick-start--run-on-colab)
- [🗂️ Project Structure](#-project-structure)
- [🛠️ Tech Stack](#-tech-stack)
- [🧭 Full Pipeline Flow](#-full-pipeline-flow)
- [⚠️ Disclaimer](#-disclaimer)
- [🙌 Acknowledgements](#-acknowledgements)

</details>

---

## 🌟 Project Overview

This project builds a **state-of-the-art Speech Emotion Recognition (SER) system** that detects human emotions directly from audio recordings. Unlike basic approaches that use only raw MFCCs with a simple CNN, this system uses a **multi-component deep learning pipeline** that mirrors techniques from published research:

| Component | Basic Approach | **This Project** |
|-----------|---------------|-----------------|
| Input | Flat MFCC vector | **3-Channel MFCC image** (MFCC + Δ + ΔΔ) |
| CNN | Plain blocks | **Squeeze-Excitation CNN** (channel attention) |
| Temporal | None / simple LSTM | **Multi-Head Attention + BiLSTM** |
| Pooling | Flatten | **Temporal Attention Pooling** (learned weights) |
| Loss | CrossEntropy | **Focal Loss** (γ=2.0, α=0.25) |
| Augmentation | None | **SpecAugment + Time Stretch + Pitch Shift + Noise** |
| Evaluation | Accuracy only | **CM + Per-class ROC-AUC + t-SNE Embeddings** |
| Deployment | None | **Advanced 6-Tab Gradio App** |

---

## 🔬 What Makes This Advanced

<table>
<tr>
<td width="50%">

### 🏗️ Architecture Innovations
- ✅ **3-Channel Spectrogram** — MFCC + ΔMFCC + ΔΔMFCC stacked like RGB
- ✅ **SE Blocks** — Channel-wise attention after every CNN block
- ✅ **Multi-Head Self-Attention** (4 heads, key_dim=64)
- ✅ **Bidirectional LSTM** — reads patterns forward AND backward
- ✅ **Temporal Attention Pooling** — soft importance weights over time
- ✅ **Residual Connections** — prevents vanishing gradients
- ✅ **Layer Normalization** — stable Transformer-style training

</td>
<td width="50%">

### 🎯 Training Innovations
- ✅ **Focal Loss** — focuses on hard-to-classify emotions
- ✅ **Cosine Annealing LR** with warm restarts
- ✅ **Gradient Clipping** (norm=1.0) — prevents LSTM explosion
- ✅ **SpecAugment** — time + frequency masking
- ✅ **4 Augmentation Methods** — stretch, pitch, noise, spec
- ✅ **Early Stopping** (patience=18) — prevents overfitting
- ✅ **Checkpoint Saving** — always keeps best weights

</td>
</tr>
<tr>
<td width="50%">

### 📊 Evaluation Innovations
- ✅ **t-SNE Embedding Space** — visualize 256D → 2D clusters
- ✅ **Per-class ROC-AUC** — individual curve for each emotion
- ✅ **Normalized Confusion Matrix** — see which emotions confuse
- ✅ **Training Curves** — accuracy + loss with best epoch marker
- ✅ **Per-class F1 Report** — precision, recall, F1 per emotion

</td>
<td width="50%">

### 🌐 Deployment Innovations
- ✅ **6-Tab Gradio App** — complete interactive interface
- ✅ **Live Audio Visualization** — waveform + mel + MFCC on upload
- ✅ **Prediction History** — tracks all session predictions
- ✅ **Top-5 Predictions** — not just the top-1 guess
- ✅ **Confidence Level** — Very Confident / Confident / Uncertain

</td>
</tr>
</table>

---

## 🎭 The 8 Emotion Classes

<div align="center">

| # | Emoji | Emotion | Voice Characteristics | RAVDESS Code |
|---|-------|---------|----------------------|--------------|
| 1 | 😐 | **Neutral** | Flat tone, steady energy, no extremes | 01 |
| 2 | 😌 | **Calm** | Slow, controlled, low energy variation | 02 |
| 3 | 😊 | **Happy** | Fast speech, high pitch, energetic bursts | 03 |
| 4 | 😢 | **Sad** | Slow, low pitch, falling intonation | 04 |
| 5 | 😠 | **Angry** | Loud, fast, high energy, harsh transitions | 05 |
| 6 | 😨 | **Fearful** | Trembling, irregular patterns, rising pitch | 06 |
| 7 | 🤢 | **Disgust** | Pulled-back vocal quality, low, tense | 07 |
| 8 | 😲 | **Surprised** | Sudden pitch jump, short bursts, sharp onset | 08 |

</div>

> **Why 8 emotions?** RAVDESS uses the Geneva Emotion Wheel subset — these 8 cover the primary human emotions with maximum acoustic contrast between them.

---

## 📊 Dataset — RAVDESS

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│          RAVDESS — Ryerson Audio-Visual Database                │
│       of Emotional Speech and Song                              │
├────────────────────┬────────────────────────────────────────────┤
│  Total Files       │  1,440 speech audio files                 │
│  Actors            │  24 professional (12 male + 12 female)    │
│  Format            │  WAV, 48kHz stereo (resampled to 22050Hz) │
│  Duration per clip │  ~3 seconds average                        │
│  Emotions          │  8 classes                                 │
│  Intensity         │  Normal + Strong (per emotion)            │
│  After Augment     │  ~2,880 total samples (2× original)       │
│  License           │  Creative Commons — free for research     │
└────────────────────┴────────────────────────────────────────────┘
```

</div>

### RAVDESS Filename Convention

```
03  -  01  -  05  -  01  -  02  -  01  -  12  .wav
│      │      │      │      │      │      └── Actor number (01–24)
│      │      │      │      │      └───────── Repetition (01–02)
│      │      │      │      └──────────────── Statement (01–02)
│      │      │      └─────────────────────── Intensity (01=Normal, 02=Strong)
│      │      └────────────────────────────── Emotion Code (01–08) ← WE USE THIS
│      └───────────────────────────────────── Vocal Channel (01=Speech)
└──────────────────────────────────────────── Modality (03=Audio Only)
```

---

## 🧠 Model Architecture

### Full Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    INPUT  (batch, 40, 130, 3)                            │
│          [40 MFCC coefficients × 130 time frames × 3 channels]          │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
            ╔════════════════╪════════════════╗
            ║   SE-CNN BACKBONE (3 Blocks)    ║
            ║                                 ║
            ║  ┌─────────────────────────┐    ║
            ║  │   SE-CNN Block 1        │    ║
            ║  │  Conv2D(64, 3×3)        │    ║
            ║  │  BatchNorm + ReLU       │    ║
            ║  │  ┌─── SE Block ───┐     │    ║
            ║  │  │ GlobalAvgPool  │     │    ║
            ║  │  │ Dense(4,relu)  │     │    ║
            ║  │  │ Dense(64,sig.) │     │    ║  Channel weights learned here
            ║  │  │ Reshape+Scale  │     │    ║
            ║  │  └────────────────┘     │    ║
            ║  │  MaxPool(2×2) → Dropout │    ║
            ║  │  Output: (20, 65, 64)   │    ║
            ║  └─────────────────────────┘    ║
            ║             ↓                   ║
            ║  ┌─────────────────────────┐    ║
            ║  │   SE-CNN Block 2        │    ║
            ║  │  Conv2D(128, 3×3)       │    ║
            ║  │  SE Block + MaxPool(2×2)│    ║
            ║  │  Output: (10, 32, 128)  │    ║
            ║  └─────────────────────────┘    ║
            ║             ↓                   ║
            ║  ┌─────────────────────────┐    ║
            ║  │   SE-CNN Block 3        │    ║
            ║  │  Conv2D(256, 3×3)       │    ║
            ║  │  SE Block + MaxPool(2×1)│    ║
            ║  │  Output: (5, 32, 256)   │    ║
            ║  └─────────────────────────┘    ║
            ╚════════════════╪════════════════╝
                             │
                    Permute (2,1,3)
                    Freq-Average Pool
                             │
                    (batch, 32, 256)   ← sequence of 32 time steps
                             │
            ╔════════════════╪════════════════╗
            ║  MULTI-HEAD SELF-ATTENTION      ║
            ║  4 heads × key_dim=64           ║
            ║  All 32 positions attend each   ║
            ║  other simultaneously           ║
            ║  + Residual Add                 ║
            ║  + LayerNormalization           ║
            ╚════════════════╪════════════════╝
                             │
            ╔════════════════╪════════════════╗
            ║   BIDIRECTIONAL LSTM            ║
            ║                                 ║
            ║  BiLSTM(128 units)   →→→        ║
            ║              ←←← BiLSTM(128)   ║  Reads forward AND backward
            ║  Output: (batch, 32, 256)       ║
            ║                                 ║
            ║  BiLSTM(64 units)    →→→        ║
            ║              ←←← BiLSTM(64)    ║
            ║  Output: (batch, 32, 128)       ║
            ╚════════════════╪════════════════╝
                             │
            ╔════════════════╪════════════════╗
            ║  TEMPORAL ATTENTION POOLING     ║
            ║                                 ║
            ║  Dense(1, tanh) → scores        ║  Which time steps matter?
            ║  Softmax(axis=1) → weights      ║  Learned soft weights
            ║  Multiply + Sum over time       ║
            ║  Output: (batch, 128)           ║
            ╚════════════════╪════════════════╝
                             │
            ╔════════════════╪════════════════╗
            ║  CLASSIFICATION HEAD            ║
            ║                                 ║
            ║  Dense(256, relu) ← EMBEDDING   ║  ← t-SNE extracted here
            ║  BatchNorm + Dropout(0.40)      ║
            ║  Dense(128, relu)               ║
            ║  Dropout(0.30)                  ║
            ║  Dense(8, softmax)              ║
            ╚════════════════╪════════════════╝
                             │
                    8 Emotion Probabilities
              [neutral, calm, happy, sad, angry,
               fearful, disgust, surprised]
```

---

## 🎛️ 3-Channel Input Design

> **The Key Innovation:** Instead of a flat feature vector, each audio clip becomes a **3-channel spectrogram image** — just like how images have RGB channels, our audio has MFCC channels.

```
Audio File (3 seconds)
        ↓
librosa.load(sr=22050) → 66,150 raw samples
        ↓
┌────────────────────────────────────────────────────┐
│                                                    │
│  Channel 0 — MFCC          (40 × 130)             │
│  ┌──────────────────────────────────────────────┐  │
│  │ ▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░│  │
│  │ "WHAT the voice sounds like right now"       │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Channel 1 — Δ MFCC        (40 × 130)             │
│  ┌──────────────────────────────────────────────┐  │
│  │ ░░▓▓░░▓▓░░░░▓▓░░░░▓▓░░░░▓▓░░░░▓▓░░░░▓▓░░│  │
│  │ "HOW FAST the voice is changing"             │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Channel 2 — ΔΔ MFCC       (40 × 130)             │
│  ┌──────────────────────────────────────────────┐  │
│  │ ▓░▓░▓░▓░░░▓░▓░▓░░░▓░▓░▓░░░▓░▓░▓░░░▓░▓░│  │
│  │ "The ACCELERATION of change (dynamics)"      │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Stack → (40, 130, 3)  ← 3-channel image           │
│  Normalize per channel → zero-mean, unit-variance  │
└────────────────────────────────────────────────────┘
```

**Why MFCC + Δ + ΔΔ works better than MFCC alone:**

```
😊 Happy voice:
   MFCC  = high energy upper bands (bright, full)
   Δ MFCC = rapid changes (fast emotional transitions)
   ΔΔ MFCC = high acceleration (bursts of energy)

😢 Sad voice:
   MFCC  = low energy, compressed spectrum
   Δ MFCC = very slow changes (monotone delivery)
   ΔΔ MFCC = near-zero acceleration (no energy bursts)

The model sees all three simultaneously → richer understanding
```

---

## ⚡ Data Augmentation Pipeline

Every audio file creates TWO training samples (original + augmented), effectively **doubling the dataset** to ~2,880 samples.

```
┌──────────────────────────────────────────────────────────┐
│              AUGMENTATION PIPELINE                       │
│                                                          │
│  Stage 1 — Waveform Level (before feature extraction)   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Randomly choose ONE of:                         │   │
│  │  ├── Time Stretch    (rate ∈ [0.85, 1.15])      │   │
│  │  │   → ±15% faster or slower speech             │   │
│  │  ├── Pitch Shift     (steps ∈ [-2, +2])         │   │
│  │  │   → ±2 semitones up or down                  │   │
│  │  ├── Gaussian Noise  (amplitude = 0.003)         │   │
│  │  │   → Simulates recording imperfections        │   │
│  │  └── None            (keep original)            │   │
│  └──────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Stage 2 — Spectrogram Level (SpecAugment)              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Time Masking  (×2 iterations):                  │   │
│  │  → Zero out 0–15 random consecutive time frames  │   │
│  │  Frequency Masking (×2 iterations):              │   │
│  │  → Zero out 0–8 random consecutive freq bands    │   │
│  │  Result: Model can't rely on any single region   │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 🔥 Focal Loss — Why Not CrossEntropy?

```
Standard CrossEntropy:    Loss = -log(p_t)
Focal Loss:               Loss = -α(1 - p_t)^γ · log(p_t)

Where:
  p_t = predicted probability for the correct class
  γ   = focusing parameter (we use 2.0)
  α   = balance factor (we use 0.25)
```

**Intuition:**

```
If model is CONFIDENT and CORRECT  (p_t = 0.95):
  CE Loss contribution   = -log(0.95) = 0.051  ← still included
  Focal Loss contribution = (1-0.95)^2 × 0.051 = 0.00013  ← almost zero!

If model is WRONG and CONFUSED  (p_t = 0.05):
  CE Loss contribution   = -log(0.05) = 3.0
  Focal Loss contribution = (1-0.05)^2 × 3.0 = 2.7  ← still large!

Result: Easy examples barely affect training.
        Hard examples (fearful vs surprised) dominate the gradient.
        Model is FORCED to get better at the hard cases.
```

---

## 📈 Training Strategy

<div align="center">

| Component | Setting | Reasoning |
|-----------|---------|-----------|
| **Optimizer** | Adam + `clipnorm=1.0` | Prevents LSTM gradient explosion |
| **LR Schedule** | Cosine Annealing + Warm Restarts | Escapes local minima; better convergence |
| **Initial LR** | `1e-3` | Standard starting point for Adam |
| **Min LR** | `1e-6` | Don't go so low training stalls |
| **Loss Function** | Focal Loss (`γ=2.0, α=0.25`) | Focus on hard emotions |
| **Early Stopping** | patience=18, `val_accuracy` | Stop when learning plateaus |
| **Checkpoint** | Save best by `val_accuracy` | Always recover from overfitting |
| **Batch Size** | 32 | Good balance of gradient quality vs speed |
| **Max Epochs** | 100 | EarlyStopping fires around epoch 40–70 |

</div>

### Cosine Annealing with Warm Restarts

```
LR
 ↑
1e-3 ─╮     ╭─╮         ╭─╮
       ╰────╯  ╰─────────╯  ╰─── ...
              ↑           ↑
           Restart      Restart
           (× t_mul=1.5 longer each time)

Why? Stuck in a bad local minimum → restart with high LR → explore → find better minimum
```

---

## 📊 Results & Evaluation

### Training Metrics

<div align="center">

| Metric | Value |
|--------|-------|
| Expected Test Accuracy | **60–75%** *(8-class SER is genuinely hard)* |
| Best Validated Approach | SE-CNN + MHA + BiLSTM |
| t-SNE Cluster Quality | Clearly separable emotion clusters |
| Easiest Emotions | 😊 Happy · 😐 Neutral · 😠 Angry |
| Hardest Pair | 😨 Fearful ↔ 😲 Surprised *(similar acoustics)* |

</div>

> **Context:** Human judges achieve ~70% on RAVDESS 8-class. A model achieving 65–75% is performing at near-human level for this task. This is NOT a failure — it reflects the genuine difficulty of speech emotion recognition.

### What t-SNE Tells Us

```
GOOD t-SNE result:                  BAD t-SNE result:
😊 😊 😊                             😊 😢 😠 😊
   😊 😊      😠 😠 😠                 😠 😊 😢 😠
😢 😢 😢         😠                   😢 😠 😊 😢
   😢                                 (everything mixed up)
→ Model learned meaningful          → Model failed to learn
  emotion representations             emotion structure
```

### Evaluation Outputs Generated

```
/tmp/
├── task2_eda.png              ← Dataset exploration (6 panels)
├── task2_feature_maps.png     ← 3-channel feature maps per emotion
├── task2_training_curves.png  ← Accuracy + Loss over epochs
├── task2_confusion_matrix.png ← Normalized 8×8 confusion matrix
├── task2_roc_curves.png       ← Per-class ROC-AUC (2×4 grid)
└── task2_tsne.png             ← 256D → 2D embedding visualization
```

---

## 🖥️ Gradio Web App — 6 Tabs

<div align="center">

| Tab | Icon | What You Can Do |
|-----|------|----------------|
| **Analyze Speech** | 🎤 | Upload WAV/MP3/FLAC or record via microphone → get emotion + confidence + visualization |
| **Session History** | 📝 | Track all predictions made during the session with refresh button |
| **Model Performance** | 📊 | Training curves + Confusion Matrix + Per-class ROC-AUC curves |
| **Embedding Space** | 🔭 | Interactive t-SNE plot of learned emotion clusters (256D → 2D) |
| **Data Analysis** | 📈 | RAVDESS EDA + 3-channel feature map visualizations |
| **Architecture** | ⚙️ | Full model diagram + loss function math + training strategy table |

</div>

### Live Prediction Output Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎙️  EMOTION RECOGNITION RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  😊  Emotion     :  HAPPY
  📊  Confidence  :  78.4%  (Very Confident)

  🔝 Top-5 Predictions:
  1. 😊 happy        ████████████████    78.4%
  2. 😌 calm         ███░░░░░░░░░░░░░░   14.2%
  3. 😐 neutral      █░░░░░░░░░░░░░░░░    4.1%
  4. 😲 surprised    ░░░░░░░░░░░░░░░░░    2.5%
  5. 😢 sad          ░░░░░░░░░░░░░░░░░    0.8%

  🤖 Architecture  :  SE-CNN → MHA → BiLSTM
  🎯 Test Accuracy :  71.30%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠️  Educational project only.
```

---

## 🚀 Quick Start — Run on Colab

### Option 1: One-Click Colab Launch

```
1. Click → "Open in Colab" badge at top of this README
2. Runtime → Change Runtime Type → T4 GPU  ← IMPORTANT
3. Runtime → Run All
4. Wait ~8–15 minutes for training
5. Click the public Gradio URL that appears at the end
```

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CodeAlpha_EmotionRecognition
cd CodeAlpha_EmotionRecognition

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the script
python task_2_advanced_emotion_recognition.py

# 5. Open the app
# → http://localhost:7860
```

### Requirements File

```txt
tensorflow>=2.12.0
librosa>=0.10.0
soundfile>=0.12.0
gradio>=4.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## 🗂️ Project Structure

```
CodeAlpha_EmotionRecognition/
│
├── 📄 task_2_advanced_emotion_recognition.py   ← Main script (17 cells)
├── 📄 README.md                                ← This file
├── 📄 requirements.txt                         ← Python dependencies
│
├── 📁 /tmp/  (auto-generated during runtime)
│   ├── ravdess/                    ← Extracted RAVDESS audio (~440 MB)
│   ├── task2_adv_best.keras        ← Saved best model weights
│   ├── task2_scaler.pkl            ← StandardScaler (fit on train)
│   ├── task2_le.pkl                ← LabelEncoder (emotion → int)
│   ├── task2_config.pkl            ← Inference configuration dict
│   ├── task2_eda.png               ← EDA visualization
│   ├── task2_feature_maps.png      ← 3-channel feature map grid
│   ├── task2_training_curves.png   ← Training history plots
│   ├── task2_confusion_matrix.png  ← Normalized confusion matrix
│   ├── task2_roc_curves.png        ← Per-class ROC-AUC curves
│   ├── task2_tsne.png              ← t-SNE embedding visualization
│   └── task2_live_viz.png          ← Real-time audio visualization
│
└── 📁 assets/  (optional, for README images)
    └── demo_screenshot.png
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.10+ | Core development |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | 2.x | Model training |
| **Audio** | ![Librosa](https://img.shields.io/badge/-Librosa-8B5CF6?logoColor=white) | 0.10+ | MFCC extraction + augmentation |
| **Data** | ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Latest | Arrays + DataFrames |
| **ML Utils** | ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | 1.3+ | Preprocessing + metrics |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logoColor=white) ![Seaborn](https://img.shields.io/badge/-Seaborn-3498DB?logoColor=white) | Latest | Charts + confusion matrix |
| **Web App** | ![Gradio](https://img.shields.io/badge/-Gradio-FF7C00?logoColor=white) | 4.x | Interactive UI deployment |
| **Platform** | ![Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white) | T4 GPU | Training environment |

</div>

---

## 🧭 Full Pipeline Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                    COMPLETE SYSTEM PIPELINE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STEP 1: DATA                                                        ║
║  RAVDESS (~440MB) → Parse filenames → Extract emotion labels         ║
║  1,440 files × 8 emotions × 2 genders × 2 intensities               ║
║                          ↓                                          ║
║  STEP 2: FEATURE EXTRACTION                                          ║
║  For each .wav file:                                                  ║
║  → librosa.load(sr=22050, duration=3s)                               ║
║  → MFCC(40) + delta(MFCC) + delta²(MFCC)                           ║
║  → Normalize per channel                                             ║
║  → Fix to 130 time frames (pad/truncate)                            ║
║  → Stack → (40, 130, 3) 3-channel image                             ║
║                          ↓                                          ║
║  STEP 3: AUGMENTATION                                                ║
║  Each file → 2 samples (original + augmented)                        ║
║  Waveform: stretch / pitch / noise                                   ║
║  Spectrogram: SpecAugment (time + freq masking)                      ║
║  Total: ~2,880 samples                                               ║
║                          ↓                                          ║
║  STEP 4: PREPROCESSING                                               ║
║  Train(70%) / Val(15%) / Test(15%) — stratified                      ║
║  StandardScaler.fit(train) → transform(all)                          ║
║  Shapes: (N, 40, 130, 3)                                             ║
║                          ↓                                          ║
║  STEP 5: MODEL TRAINING                                              ║
║  SE-CNN → Multi-Head Attention → BiLSTM → Temporal Attention         ║
║  Focal Loss + Cosine Annealing LR + EarlyStopping                   ║
║  Best model saved by val_accuracy                                    ║
║                          ↓                                          ║
║  STEP 6: EVALUATION                                                  ║
║  Test set → Confusion Matrix + ROC-AUC + t-SNE                       ║
║  Best checkpoint loaded → final metrics computed                     ║
║                          ↓                                          ║
║  STEP 7: DEPLOYMENT                                                  ║
║  Gradio 6-Tab App → live prediction + visualization                  ║
║  upload audio → extract features → scale → predict → display         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## ⚠️ Disclaimer

> This project is developed purely for **educational purposes** as part of the **CodeAlpha Machine Learning Internship**.
>
> - It is **not** a clinical or commercial emotion recognition system
> - Results may vary based on recording quality, speaker accent, and background noise
> - For production deployment, larger and more diverse datasets are required
> - Always consult domain experts before using AI in sensitive applications

---

## 🙌 Acknowledgements

- **[CodeAlpha](https://www.codealpha.tech)** — For designing impactful real-world internship projects
- **[RAVDESS](https://zenodo.org/record/1188976)** — Livingstone & Russo (2018), Zenodo
- **[Librosa](https://librosa.org)** — Brian McFee et al. — Audio analysis library
- **[TensorFlow/Keras](https://tensorflow.org)** — Google Brain Team
- **[Gradio](https://gradio.app)** — Abubakar Abid et al. — Web app framework
- **[SpecAugment](https://arxiv.org/abs/1904.08779)** — Park et al., 2019
- **[Focal Loss](https://arxiv.org/abs/1708.02002)** — Lin et al., 2017 (RetinaNet)
- **[Squeeze-Excitation Networks](https://arxiv.org/abs/1709.01507)** — Hu et al., 2018

---

## 📬 Contact

<div align="center">

| Platform | Link |
|----------|------|
| 🌐 **CodeAlpha** | [www.codealpha.tech](https://www.codealpha.tech) |
| 📧 **Email** | services@codealpha.tech |
| 💬 **WhatsApp** | +91 9336576683 |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>

**Made with ❤️ by Taj Wali Khan · CodeAlpha ML Internship — Task 2**

*"The best models don't just classify — they understand."*

⭐ **Star this repo** if it helped you learn something new!

</div>
