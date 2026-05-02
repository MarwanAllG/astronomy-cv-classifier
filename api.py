from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Recreate the same model architecture used in training
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 7)

# Load the saved best weights
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Replace these labels with your actual class names in order
class_names = [
    "asteroid",
    "black hole",
    "comet",
    "galaxy",
    "nebula",
    "planet",
    "star",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "Invalid image file", "details": str(exc)})

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        score, predicted = torch.max(probabilities, dim=0)

    prediction = class_names[predicted.item()]
    confidence = float(score.item())

    return {"prediction": prediction, "confidence": confidence}