# 🜂 DEHA-Net-Replication — Dual-Encoder Hard Attention Segmentation Framework

This repository provides a **PyTorch-based research replication** of  
**DEHA-Net: A Dual-Encoder-Based Hard Attention Network with an Adaptive ROI Mechanism for Lung Nodule Segmentation**,  
implemented as a **theory-faithful medical segmentation framework**.

The project translates the paper’s **dual-encoder topology, hard attention gates, adaptive ROI mechanism, and tri-planar consensus inference**
into a clean, modular, and extensible research codebase.

- Enables **high-precision lung nodule segmentation from CT slices** 🫁  
- Implements **global–local dual-encoder representation learning** 🧠  
- Integrates **hard attention gating for region-focused feature fusion** 🜄  
- Employs **adaptive ROI refinement for coarse-to-fine segmentation** 🜁  
- Supports **tri-planar consensus inference (axial, coronal, sagittal)** 🜃  

**Paper reference:**  [DEHA-Net: Dual-Encoder Hard Attention Network with Adaptive ROI for Lung Nodule Segmentation (2023)](https://www.mdpi.com/1424-8220/23/4/1989) 📄


---

## 🝆 Overview — Dual-Encoder Hard Attention Segmentation Pipeline

🜂 Global Encoder → 🜄 Local Encoder → 🜁 Hard Attention → 🜃 Adaptive ROI → 🜀 Decoder → 🫁 Segmentation Mask

The core idea:

> Lung nodules are small, heterogeneous, and easily confused with vessels and bronchi.  
> Accurate segmentation requires both global anatomical context and local fine-grained focus.

Instead of relying on a single encoder, DEHA-Net performs **dual-stream feature extraction**:

$$
I \longrightarrow \hat{Y}
$$

where the model learns a slice-wise mapping

$$
f_\theta : \mathbb{R}^{H \times W} \rightarrow \mathbb{R}^{H \times W}
$$

and produces a dense segmentation mask $\hat{Y}$ from a CT slice $I$.

The architecture follows a **dual-encoder + attention-gated decoder design** enriched with an  
**Adaptive Region of Interest (A-ROI) refinement mechanism**.

---

## 🧠 Architectural Principle — DEHA-Net

The network consists of two parallel encoders:

- **Global Encoder** 🜂 — learns anatomical context and coarse localization  
- **Local Encoder** 🜄 — learns fine-grained nodule appearance from ROI patches  

At each decoding stage, features are fused using **Hard Attention Gates**:

$$
\alpha = \sigma(\psi(\text{ReLU}(W_g g + W_l l)))
$$

$$
\hat{l} = \alpha \odot l
$$

where  
$g$ is the global feature,  
$l$ is the local feature,  
and $\alpha$ is the spatial attention mask.

This forces the decoder to focus only on **nodule-relevant regions**.

---

## 🜁 Adaptive ROI Mechanism (A-ROI)

The Adaptive ROI module refines segmentation via a coarse-to-fine strategy.

Given an initial probability map $P$:

$$
ROI = \{ p \mid P(p) > R_T \}
$$

A bounding box is extracted around high-confidence pixels and expanded with a margin.

This ROI is then cropped and re-fed into the local encoder for refined prediction.

This mimics a **radiologist zooming into a suspicious region**.

---

## 🜃 Tri-Planar Consensus Inference

CT volumes are interpreted along three anatomical planes:

- Axial  
- Coronal  
- Sagittal  

Each view is segmented independently and fused via consensus:

$$
\hat{Y} = \frac{Y_{axial} + Y_{coronal} + Y_{sagittal}}{3}
$$

This provides **3D spatial consistency** from a 2D network.

---

## 🔬 Mathematical Formulation

Let the input CT slice be

$$
I \in \mathbb{R}^{H \times W}
$$

The network learns a pixel-wise classifier:

$$
p(y_{ij} \mid I) = \sigma(f_\theta(I)_{ij})
$$

Training is performed using Dice loss:

$$
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}
$$

$$
\mathcal{L}_{dice} = 1 - \text{Dice}
$$

where  
$P$ is the predicted mask and  
$G$ is the ground-truth mask.

This directly optimizes spatial overlap — critical for medical segmentation.

---

## 🧪 What the Model Learns

- To distinguish nodules from vessels and bronchi 🜇  
- To focus attention only on suspicious regions 🜄  
- To refine segmentation via adaptive zooming 🜁  
- To preserve fine boundary geometry 🝀  
- To reason across 3D anatomy using multi-view consensus 🜃  

Segmentation becomes a **context-aware, attention-guided reasoning task**.

---

## 📦 Repository Structure

```bash
DEHA-Net-Replication/
├── src/
│
│   ├── model/
│   │   ├── encoders.py           # Global + Local encoder
│   │   ├── attention.py          # Hard Attention Gate (paper equation)
│   │   ├── decoder.py            # Decoder blocks
│   │   └── deha_net.py           # Full DEHA-Net assembly
│   │
│   ├── roi/
│   │   ├── adaptive_roi.py       # A-ROI algorithm (paper Section 3.3)
│   │   └── roi_utils.py          # Bounding box, margin, propagation
│   │
│   ├── consensus/
│   │   └── consensus_module.py   # Axial + Sagittal + Coronal fusion
│   │
│   ├── dataset/
│   │   └── lidc_loader.py
│   │
│   ├── pipeline/
│   │   └── inference_pipeline.py # Paper inference flow
│   │
│   ├── visualization/
│   │   └── overlay.py            # CT + ROI + Mask overlay
│   │
│   └── config.py
│
├── requirements.txt
└── README.md
```
---


## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
