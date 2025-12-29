# NeuroCanvas: AI Art from Brainwaves

**Multi‑Modal AI System | EEG Signals + Text → Unique Art Generation**

---

## 🌟 Overview

NeuroCanvas is an end-to-end multi-modal AI project that fuses real EEG brainwave signals with natural-language prompts to generate unique AI artworks that visually represent mental state and imagination. The system combines EEG preprocessing, a cross-modal fusion network, and Stable Diffusion — all exposed via a real-time Gradio interface for demos and exploration.

---

## 🧩 Key Features

* **EEG Signal Processing** — Robust extraction and preprocessing of EEG features from real datasets.
* **Multi-Modal Fusion Model** — Learns alignment between brain-state embeddings and text embeddings.
* **Stable Diffusion Integration** — High-quality image generation conditioned on fused embeddings.
* **Custom Neural Network** — Lightweight networks (~500K parameters) tailored for EEG+text fusion.
* **Interactive Gradio UI** — Real-time controls, visualizations, and generation pipeline.
* **Deployment Ready** — Docker configs and multiple deployment pathways (Hugging Face Spaces, Cloud Run, SageMaker).
* **Well-documented & Tested** — Clear code organization, logging, and error handling.

---

## 🧠 System Architecture

```
EEG Signal  +  Text Prompt
      │              │
      ▼              ▼
 ┌────────┐    ┌────────────┐
 │ EEGNet │    │ TextEncoder│
 └────┬───┘    └─────┬──────┘
      │               │
      ▼               ▼
     └──────▶ Fusion Network ◀──────┘
                  │
                  ▼
        🎨 Stable Diffusion
                  │
                  ▼
          Generated Artwork
```

---

## ⚙️ Tech Stack

**Programming:** Python, PyTorch

**Deep Learning:** Transformers, Stable Diffusion

**Data Handling:** NumPy, Pandas, scikit-learn

**Visualization:** Matplotlib (note: seaborn used only for exploratory plots)

**Interface:** Gradio

**Deployment:** Hugging Face Spaces, Google Cloud Run, AWS SageMaker, Heroku

---

## 📂 Project Structure

```
neurocanvas/
├── app.py                 # Main Gradio application entry point
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                 # Environment variables (not committed)
├── .gitignore           # Git ignore rules
├── config/
│   └── settings.py      # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py    # Dataset loading & Kaggle integration
│   │   └── processor.py # EEG preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion.py    # EEG-Text fusion model
│   │   └── generator.py # Art generation pipeline
│   └── ui/
│       ├── __init__.py
│       └── interface.py # Gradio UI components
├── data/
│   ├── raw/            # Raw EEG data
│   ├── processed/      # Processed features
│   └── models/         # Saved models
└──   Nuerocanvas.ipynb      # Jupyter notebooks for experimentation

---

## 📊 Model Performance & Metrics

* **Parameters:** ~500K
* **Training Dataset:** Public EEG datasets (example: Kaggle EEG sets)
* **Inference Time:** ~3–5 seconds per image (depends on hardware)
* **Validation:** Fusion metrics demonstrate strong alignment between brain-state and prompt semantics (report metric plots in `/results`)

---

## 🧠 Example Prompts & Settings

| Brain Activity Level | Art Prompt                                | Steps | Guidance |
| -------------------- | ----------------------------------------- | ----: | -------: |
| High (Alert)         | Neural networks glowing with electricity  |    30 |      7.5 |
| Medium (Relaxed)     | Peaceful zen garden in the mind           |    30 |      7.5 |
| Low (Drowsy)         | Dreamlike clouds of consciousness         |    30 |      7.5 |
| High (Alert)         | Cyberpunk brain interface with neon lines |    30 |      8.0 |
| Medium (Relaxed)     | Abstract thoughts flowing like water      |    30 |      7.0 |

> Tip: Map EEG-derived scalar features (e.g., alpha/beta ratios, engagement scores) to conditioning strength or style parameters for dynamic variation.

---

## 🚀 How to Run (Local)

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Prepare data (example):

```bash
python data/fetch_data.py
python data/preprocess.py
```

3. Start the Gradio demo:

```bash
python app.py
```

---

## 🧭 Deployment Options

* **Hugging Face Spaces** — Easiest for portfolio demos; ideal for CPU/GPU-based web demos.
* **Google Cloud Run** — Containerized, autoscaling deployments.
* **AWS SageMaker** — For production model serving and monitoring.
* **Heroku** — Quick prototype hosting (less suited for GPU workloads).

---

## 📈 Future Enhancements

* Integrate real EEG hardware (Muse, OpenBCI) for live captures.
* Dynamic style transfer based on continuous brain-state trajectories.
* User gallery, sharing, and versioning of generated art.
* Mobile client for on-device viewing and lightweight generation.
* Video generation from EEG time-series sequences.

---

## 🧑‍💻 Interview Talking Points

* Architecture rationale: why fuse EEG with text and how alignment improves semantics.
* Data challenges: artifact removal, subject variability, label sparsity.
* Training details: loss functions, scheduling, multimodal contrastive or alignment objectives.
* Deployment considerations: latency, GPU vs CPU inference, user privacy for EEG data.
* Demo walkthrough: show preprocessing → fusion → Stable Diffusion pipeline live on Gradio.

---

## 🧾 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

---

## 🧠 About the Creator

DEMO : " https://drive.google.com/file/d/1GmkyCU0HdTKcqvbXvawgK4Rhx-OBEVF6/view?usp=drive_link "

Created by **[kamshetty varun]**, AI/ML Engineer passionate about brain-computer interfaces and creative AI systems.

---

⭐ If you find this project inspiring, don’t forget to star the repo!

---


