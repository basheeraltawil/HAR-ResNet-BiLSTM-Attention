# <div align="center"> <strong>Multi-Head Attention-Based Framework with Residual Network for Human Action Recognition</strong> </div>

![Model Architecture](./images/model.png)

---

##  Overview

This repository provides the full implementation of our human action recognition framework described in the paper:

**"Multi-Head Attention-Based Framework with Residual Network for Human Action Recognition"**  
 *Sensors, 2025*  
🔗 [DOI: 10.3390/s25092930](https://www.mdpi.com/1424-8220/25/9/2930)

Our framework combines:
- ✅ **ResNet-18** for extracting spatial features
- 🔁 **BiLSTM** for modeling temporal dynamics
- 🎯 **Multi-Head Attention** for motion focus
- 🌀 **Motion-based Frame Selection** via optical flow

---

## 📂 Project Structure

```bash
HAR-ResNet-BiLSTM-Attention/
├── configs/                  # Configuration files
├── extracted_features/      # Extracted Features from row dataset
├── checkpoints/             # Saved model checkpoints
├── logs/                    # Training/validation logs and plots
├── images/                  # Visuals (model diagram, etc.)
│   └── model.png
├── scripts/
│   ├── extract_feature_nm.py    # Feature extraction (no motion consideration)
│   ├── extract_feature_wm.py    # Feature extraction (with motion consideration)
│   ├── lstm_model.py            # BiLSTM + Attention model definition
│   ├── test_on_pc_camera.py     # Test model with webcam input
│   ├── test_on_ucf101_data.py   # Evaluate model on full UCF101 test set
│   └── training_model_bilstm.py # Train model with extracted features
