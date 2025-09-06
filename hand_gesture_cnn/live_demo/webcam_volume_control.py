import cv2
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import osascript  # macOS system control
import time
import subprocess

# Add parent directory to path for model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import HandGestureCNN

# Load model
model = HandGestureCNN(num_classes=6)
model.load_state_dict(torch.load("checkpoints/best_model_v2.pth", map_location="cpu"))
model.eval()

# Gesture-to-action mapping
idx_to_class = {0: 'dislike', 1: 'fist', 2: 'like', 3: 'ok', 4: 'palm', 5: 'rock'}

# Initial volume level
volume_level = 50
osascript.osascript(f"set volume output volume {volume_level}")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Action functions
def set_volume(level):
    level = max(0, min(level, 100))
    osascript.osascript(f"set volume output volume {level}")
    #subprocess.call(["say", "volume changed"])

def play():
    osascript.osascript('tell application "Music" to play')
    subprocess.call(["say", "playing"])

def pause():
    osascript.osascript('tell application "Music" to pause')
    subprocess.call(["say", "paused"])

def replay():
    osascript.osascript('tell application "Music" to set player position to 0')
    # No speech for replay (OK gesture does nothing now)

def stop():
    osascript.osascript('tell application "Music" to stop')
    subprocess.call(["say", "stopped"])

# Cooldown management
last_action_time = time.time()
cooldown_seconds = 3  # Time to wait before repeating action

# Open webcam
cap = cv2.VideoCapture(0)
print("🎵 Webcam volume control started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    input_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        gesture = idx_to_class[pred]

    # Display gesture
    cv2.putText(frame, f"Gesture: {gesture}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 7)
    cv2.imshow("Gesture Volume Control", frame)

    current_time = time.time()
    if current_time - last_action_time >= cooldown_seconds:
        if gesture == "rock":
            play()
        elif gesture == "palm":
            pause()
        elif gesture == "like":
            volume_level += 30
            set_volume(volume_level)
        elif gesture == "dislike":
            volume_level -= 30
            set_volume(volume_level)
        elif gesture == "ok":
            pass  # Do nothing
        elif gesture == "fist":
            stop()

        last_action_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
