# 📸 Webcam Hand Gesture Predictor

This script (`webcam_predict.py`) runs a **real-time hand gesture classifier** using your webcam and a pre-trained CNN model built with PyTorch.

## 🎯 Purpose
- Live detection of hand gestures
- Visual overlay of gesture label on camera feed
- Works locally using your laptop’s webcam

## 🤖 Recognized Gestures
| Class Index | Gesture   |
|-------------|-----------|
| 0           | dislike   |
| 1           | fist      |
| 2           | like      |
| 3           | ok        |
| 4           | palm      |
| 5           | rock      |

## 🛠 Requirements
```bash
pip install torch torchvision opencv-python
```

## 🚀 How to Run
From the root of your project:
```bash
python live_demo/webcam_predict.py
```

### 📦 Make Sure You Have:
- `checkpoints/best_model.pth` saved
- `models/model.py` containing `HandGestureCNN`

## 📂 How It Works
1. Loads a pre-trained model from disk
2. Captures live frames using OpenCV
3. Applies PyTorch transforms
4. Predicts the gesture and overlays it on the video feed

## 🔁 Useful For
- Model sanity checks
- Debugging gesture classification before integrating actions

---

🛠 Want more? Check out [`webcam_volume_control.py`](./webcam_volume_control.py) to control your system with gestures!