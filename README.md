# <div align="center"> <strong>Multi-Head Attention-Based Framework with Residual Network for Human Action Recognition</strong> </div>

![Model Architecture](./images/model.png)

---

## 🚀 Overview

This repository provides the full implementation of our human action recognition framework described in the paper:

**"Multi-Head Attention-Based Framework with Residual Network for Human Action Recognition"**  
📚 *Sensors, 2025*  
🔗 [DOI: 10.3390/s25092930](https://www.mdpi.com/1424-8220/25/9/2930)

### Key Components:
- **ResNet-18** for spatial feature extraction.
- **BiLSTM** for learning temporal dependencies.
- **Multi-Head Attention** for focusing on significant motion cues.
- **Motion-based Frame Selection** using optical flow to reduce redundancy.

---

## 📂 Project Structure

```bash
HAR-ResNet-BiLSTM-Attention/
├── configs/                  # Configuration files
├── extracted_features/      # ResNet-based features
├── checkpoints/             # Saved model checkpoints
├── logs/                    # Training/validation logs and plots
├── images/                  # Visuals (model diagram, etc.)
│   └── model.png
├── scripts/                 # Scripts for each pipeline stage
│   ├── extract_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── test_model.py
├── LICENSE                  # MIT License
├── README.md                # Documentation
└── .gitattributes           # Git LFS configuration
