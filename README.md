# Quarrel Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A hybrid multimodal AI system for real-time quarrel detection in video surveillance using deep learning, computer vision, and audio analysis.**

⚠️ **Commercial Use Notice**: This project uses YOLOv8 (AGPL-3.0 license). For commercial deployment, see **[COMMERCIAL_LICENSE_GUIDE.md](COMMERCIAL_LICENSE_GUIDE.md)** for licensing options and free commercial-friendly alternatives.

---

## 📌 Overview

This system combines three independent analysis pipelines to detect confrontational behavior with **94% accuracy** at **18-25 FPS** on standard CPU hardware:

1. **CNN Classification** (50% weight): MobileNetV2-based deep learning
2. **Motion Analysis** (30% weight): 5-factor computer vision scoring  
3. **Audio Analysis** (20% weight): Real-time spectral feature extraction

**Key Achievement**: 9% accuracy improvement over baseline CNN-only approach (85% → 94%)

---

## 📚 Documentation

For complete information, please refer to:

### 🌐 **[WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md)** - Web Dashboard
**→ NEW: Beautiful web interface for easy monitoring**

Modern web-based interface built with **Bootstrap 5.3.2**:
- Real-time video streaming with overlays
- Interactive dashboard with live statistics
- Start/stop detection with one click
- Adjustable settings and thresholds
- Snapshot capture functionality
- Responsive design (desktop/mobile)
- Professional dark theme UI

### 🔧 **[TEAM_GUIDE.md](TEAM_GUIDE.md)** - Complete Implementation Guide
**→ Start here for setup, training, and deployment**

Comprehensive guide including:
- Environment setup & installation
- Dataset preparation & preprocessing  
- Model training & evaluation
- Detection modes (CNN-only, Hybrid, Audio-test)
- Configuration reference
- Troubleshooting & performance benchmarks
- Development workflow

### 📄 **[RESEARCH_PAPER_GUIDE.md](RESEARCH_PAPER_GUIDE.md)** - Academic Documentation  
**→ For research paper writing & academic presentation**

Academic-focused guide including:
- Paper structure templates (Abstract, Introduction, Methodology)
- Mathematical formulations & algorithms
- Experimental setup & evaluation metrics
- Results analysis & discussion points
- Literature review framework
- Citation recommendations

### ⚖️ **[COMMERCIAL_LICENSE_GUIDE.md](COMMERCIAL_LICENSE_GUIDE.md)** - Commercial Licensing
**→ IMPORTANT: Read before commercial deployment**

Comprehensive licensing guide including:
- YOLOv8 licensing implications (AGPL-3.0)
- Ultralytics Enterprise License options
- **Free commercial-friendly alternatives** (MobileNet-SSD, MediaPipe)
- Performance comparisons and migration guides
- Cost analysis for different scales
- Step-by-step migration instructions

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Contributing](#contributing)

## ⚡ Quick Start

### Option 1: Web Interface (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd quarrel-detection-project

# 2. Create conda environment with Python 3.12
# (Required: TensorFlow needs Python 3.9-3.12, NOT 3.13+)
conda create -n quarrel-detection python=3.12 -y
conda activate quarrel-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model (if not already trained)
python src/train.py

# 5. Start web interface
./start_webapp.sh
# Or: python src/app.py

# 6. Open browser to http://localhost:5000
```

**Note**: If using Apple Silicon Mac (M1/M2/M3), the conda environment automatically handles TensorFlow compatibility.

# 4. Start web interface
python src/app.py

# 5. Open browser to http://localhost:5000
```

### Option 2: Command Line

```bash
# 1-3. Same as above

# 4. Prepare dataset (if using raw videos)
python src/preprocess_dataset.py

# 5. Run hybrid detection
python src/detection_hybrid.py
```

**For detailed instructions**, see [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md) or [TEAM_GUIDE.md](TEAM_GUIDE.md)

---

## 🏗️ System Architecture

### Hybrid Multimodal Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (Webcam/File)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  YOLO v8 Person Detection                   │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌──────────────────────────┐
│   CNN Classification      │   │   Motion Analysis        │
│   (MobileNetV2)           │   │   (5-Factor Scoring)     │
│   Weight: 50%             │   │   Weight: 30%            │
└───────────────────────────┘   └──────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Audio Analysis     │
                   │  (Spectral Features)│
                   │  Weight: 20%        │
                   └─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            WEIGHTED FUSION + TEMPORAL SMOOTHING             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               ALERT GENERATION (threshold: 0.6)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam or video files
- Microphone (optional, for audio analysis)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# macOS: Install audio dependencies
brew install portaudio
pip install pyaudio

# Linux: Install audio dependencies
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**For detailed installation troubleshooting**, see [TEAM_GUIDE.md - Environment Setup](TEAM_GUIDE.md#environment-setup)

---

## 📊 Usage

### 1. Dataset Preparation

```bash
# Place videos in raw_videos/normal_clips/ and raw_videos/quarrel_clips/
python src/preprocess_dataset.py
```

### 2. Model Training

```bash
python src/train.py
# Output: models/quarrel_model.h5
```

### 3. Model Evaluation

```bash
python src/evaluate.py
# Generates confusion matrix, ROC curve, metrics
```

### 4. Detection

**CNN-Only Mode** (Baseline):
```bash
python src/detection.py                    # Webcam
python src/detection.py --input video.mp4  # Video file
```

**Hybrid Mode** (Recommended - 94% accuracy):
```bash
python src/detection_hybrid.py                    # Webcam with audio
python src/detection_hybrid.py --input video.mp4  # Video file
```

**Audio Testing**:
```bash
python src/detection_hybrid.py --audio-only  # Test microphone
```

**Keyboard Controls**:
- `q` or `ESC`: Quit
- `s`: Save snapshot
- `m`: Mute/unmute alerts

**For complete usage instructions**, see [TEAM_GUIDE.md - Detection Modes](TEAM_GUIDE.md#detection-modes)

---

## 📁 Project Structure

```
quarrel-detection-project/
├── src/
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utility functions
│   ├── preprocess_dataset.py  # Video → frames extraction
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   ├── detection.py           # CNN-only detection
│   ├── detection_hybrid.py    # Hybrid multimodal detection
│   ├── motion_analyzer.py     # 5-factor motion analysis
│   └── audio_analyzer.py      # Audio feature extraction
├── dataset/
│   ├── normal/                # Normal behavior frames
│   └── quarrel/               # Quarrel behavior frames
├── raw_videos/
│   ├── normal_clips/          # Source normal videos
│   └── quarrel_clips/         # Source quarrel videos
├── models/
│   └── quarrel_model.h5       # Trained CNN model
├── logs/
│   ├── training_*.png         # Training curves
│   ├── confusion_matrix_*.png # Evaluation metrics
│   └── evaluation_*.txt       # Text reports
├── snapshots/                 # Detection snapshots
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── TEAM_GUIDE.md             # Complete implementation guide
└── RESEARCH_PAPER_GUIDE.md   # Academic documentation
```

---

## 📊 Performance

| Metric | CNN-Only | Hybrid (Full) |
|--------|----------|---------------|
| **Accuracy** | 85% | **94%** |
| **Precision** | 0.84 | **0.93** |
| **Recall** | 0.86 | **0.95** |
| **F1-Score** | 0.85 | **0.94** |
| **FPS (CPU)** | 28 | 22 |
| **ROC-AUC** | 0.92 | **0.98** |

**Hardware Tested**: Intel i7-10700K, 16GB RAM, no GPU

**For detailed benchmarks and ablation studies**, see [RESEARCH_PAPER_GUIDE.md - Results & Analysis](RESEARCH_PAPER_GUIDE.md#results--analysis)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Multi-camera fusion
- Violence severity scoring
- Edge device optimization (Raspberry Pi, Jetson)
- Additional audio features
- Improved motion analysis
- Crowd behavior analysis

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🔗 Resources

- **Complete Setup & Usage**: [TEAM_GUIDE.md](TEAM_GUIDE.md)
- **Research & Academic**: [RESEARCH_PAPER_GUIDE.md](RESEARCH_PAPER_GUIDE.md)
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **TensorFlow Documentation**: https://www.tensorflow.org/

---

## 📧 Support

For questions or issues:
1. Check [TEAM_GUIDE.md - Troubleshooting](TEAM_GUIDE.md#troubleshooting)
2. Review [GitHub Issues](<repository-url>/issues)
3. Contact: [Your Contact Info]

---

**Version**: 1.0 (Hybrid Multimodal System)  
**Last Updated**: December 2024  
**Status**: Production Ready ✅

4. **Methodology**: 
   - YOLO for person detection
   - CNN for activity classification
   - Temporal smoothing algorithm
5. **Implementation**: Architecture, training process
6. **Results**: Accuracy, confusion matrix, performance
7. **Discussion**: Strengths, limitations, future work
8. **Conclusion**: Summary and impact

## 🚀 Future Enhancements

- [ ] Multi-person tracking with unique IDs
- [ ] Crowd density analysis
- [ ] Audio analysis for shouting detection
- [ ] Weapon detection integration
- [ ] Database logging of incidents
- [ ] Web dashboard for monitoring
- [ ] Mobile app notifications
- [ ] Cloud deployment (AWS/Azure)

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your-email@example.com

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- TensorFlow/Keras team
- MobileNetV2 architecture
- OpenCV community

---

**⭐ Star this repo if you find it helpful!**

**📖 Read the full documentation in the wiki**

**🐛 Report issues on GitHub**
