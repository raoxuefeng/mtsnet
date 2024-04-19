## MTSNet
This folder contains the code for **MTSNet: Joint Feature Adaptation and Enhancement for Text-Guided Multi-view Martian Terrain Segmentation**.

![Framework of MTSNet](./images/framework-model.png)

## General file descriptions
- model.py - model architectures defined here.
- LTEN.py - implementation of **Local Terrain Feature Enhancement Network (LTEN)**.
- GFM.py - implementation of **Gated Fusion Module (GFM)**.
- segment_anything/* - files for Segment Anything Model , and we define the   **Terrain Context Attention Adapter Module (TCAM)** in segment_anything/modeling/adapter.py.
- clip/* - files for CLIP model.
- model_components/ema.py - code for Efficient Multi-scale Attention used in LTEN block.