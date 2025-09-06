# ✋ Hand Gesture Volume Control (MacOS)

This project is a real-time hand gesture recognition system using PyTorch and OpenCV that lets you control music playback and system volume on macOS using your webcam.

## 🎯 Features
- Real-time webcam gesture detection
- Pre-trained CNN model (`HandGestureCNN`)
- macOS volume and Music app control using `osascript`
- Cooldown logic to prevent repeated actions
- Voice feedback for gestures (except neutral "ok")

## 🤖 Gesture-to-Action Mapping
| Gesture   | Action           |
|-----------|------------------|
| `rock`    | Play             |
| `palm`    | Pause            |
| `like`    | Volume Up        |
| `dislike` | Volume Down      |
| `ok`      | Wait (no action) |
| `fist`    | Stop             |

## 🧱 Project Structure
```
hand_gesture_cnn/
├── checkpoints/              # Saved model weights
│   └── best_model.pth
├── models/
│   └── model.py              # HandGestureCNN architecture
├── live_demo/
│   ├── webcam_predict.py     # Basic live webcam classification
│   └── webcam_volume_control.py # Music and volume control logic
```

## 🛠 Requirements
- Python 3.8+
- macOS (tested on M1 Pro)
- Webcam

### 📦 Install Dependencies
```bash
pip install torch torchvision opencv-python osascript
```

## 🚀 Run the Volume Control Script
From the project root:
```bash
python live_demo/webcam_volume_control.py
```

> 🧠 Tip: Make sure your virtual environment is activated and `Music.app` is open.

## ✅ To Do / Ideas
- Add hand detection (MediaPipe) to improve accuracy
- Enable control toggling (on/off switch)
- Add GUI display or gesture confidence levels

## 🙌 Credits
- Built using PyTorch, OpenCV, and osascript
- Model trained on the HaGRID hand gesture dataset

---
🖐 Built with creativity and a MacBook by Marcos Hernandez