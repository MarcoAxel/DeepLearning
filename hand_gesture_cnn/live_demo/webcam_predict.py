# run this on bash: python live_demo/webcam_predict.py
# env dir: documents/Spring 25/CSC 360 Deep Learning/DeepLearning
# hand gesture dir: "Documents/Spring 25/CSC 360 Deep Learning/DeepLearning/DeepLearning/hand_gesture_cnn"
import cv2
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Allow importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import HandGestureCNN

# Load model
model = HandGestureCNN(num_classes=6)
model.load_state_dict(torch.load("checkpoints/best_model_v2.pth", map_location="cpu"))
model.eval()

# Class index mapping (adjust based on your dataset)
idx_to_class = {0: 'dislike', 1: 'fist', 2: 'like', 3: 'ok', 4: 'palm', 5: 'rock'}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

print("🎥 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL-compatible RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    # Preprocess image
    input_tensor = transform(image_pil).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        gesture = idx_to_class[pred]

    # Display prediction
    cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()