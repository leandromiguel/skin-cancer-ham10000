from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm


# Definição da arquitetura idêntica à usada no treino
class HybridModel(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridModel, self).__init__()

        # Backbone CNN: EfficientNet-V2-S
        self.cnn = timm.create_model(
            'tf_efficientnetv2_s.in21k_ft_in1k',
            pretrained=False,
            num_classes=0
        )

        # Backbone ViT: Swin-Tiny
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0
        )

        num_ftrs_cnn = self.cnn.num_features  # 1280
        num_ftrs_vit = self.vit.num_features  # 768

        # Cabeça classificadora idêntica ao treino
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs_cnn + num_ftrs_vit, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat_cnn = self.cnn(x)
        feat_vit = self.vit(x)
        combined = torch.cat((feat_cnn, feat_vit), dim=1)
        return self.classifier_head(combined)


CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminho absoluto do modelo — robusto independente do diretório de trabalho
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_hybrid_model.pth")

# Pipeline de transformação idêntico ao de validação/teste no treino
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Variável global do modelo
model = None


def load_model():
    global model
    try:
        model = HybridModel(num_classes=7)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Skin Cancer Classification API",
    description="API para inferência do modelo de classificação de lesões cutâneas (HAM10000).",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo não inicializado na API."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado precisa ser uma imagem válida."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())
        all_probs = {
            CLASS_NAMES[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": all_probs
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o processamento da imagem: {str(e)}"
        )