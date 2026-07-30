# Skin Cancer Classification API

API de inferência para o modelo de Deep Learning treinado no dataset HAM10000 para classificação de lesões de pele.

## Classes suportadas

| Código | Nome completo | Natureza |
|---|---|---|
| akiec | Actinic Keratosis | Pré-maligna |
| bcc | Basal Cell Carcinoma | Maligna |
| bkl | Benign Keratosis | Benigna |
| df | Dermatofibroma | Benigna |
| mel | Melanoma | Maligna |
| nv | Melanocytic Nevi | Benigna |
| vasc | Vascular Lesion | Benigna |

## Pré-requisitos

- [Docker](https://www.docker.com/) instalado e rodando
- Arquivo de pesos `best_hybrid_model.pth` na raiz da pasta `api/`

## Como executar localmente via Docker

**1. Clone o repositório:**
```bash
git clone https://github.com/SEU_USUARIO/skin-cancer-ham10000.git
cd skin-cancer-ham10000/api
```

**2. Coloque o modelo na pasta `api/`:**
```bash
# Baixe o best_hybrid_model.pth do Kaggle Output e mova para api/
mv ~/Downloads/best_hybrid_model.pth .
```

**3. Construa a imagem Docker:**
```bash
docker build -t skin-cancer-api .
```

**4. Execute o container:**
```bash
docker run -p 8000:8000 skin-cancer-api
```

**5. Acesse a documentação interativa (OpenAPI):**

## Endpoints

### `GET /`
Verifica se a API está online e se o modelo foi carregado.

**Resposta:**
```json
{
  "status": "online",
  "device": "cpu",
  "model_loaded": true
}
```

### `POST /predict`
Recebe uma imagem de lesão de pele e retorna a classificação.

**Parâmetros:**
- `file` (form-data): imagem nos formatos JPG, PNG ou JPEG

**Exemplo com curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sua_imagem.jpg"
```

**Resposta:**
```json
{
  "prediction": "mel",
  "confidence": 87.43,
  "probabilities": {
    "akiec": 0.0312,
    "bcc": 0.0215,
    "bkl": 0.0187,
    "df": 0.0098,
    "mel": 0.8743,
    "nv": 0.0341,
    "vasc": 0.0104
  }
}
```

## Arquitetura do modelo

O modelo utiliza uma arquitetura dual-backbone:
- **EfficientNet-V2-S** — extração de features locais (texturas e padrões da lesão)
- **Swin-Tiny (ViT)** — extração de features globais (contexto espacial)
- **Feature Fusion** — concatenação dos vetores `[Batch, 1280]` e `[Batch, 768]`
- **Classifier Head** — Linear(2048→512) → BatchNorm → SiLU → Dropout(0.4) → Linear(512→7)

## Estrutura do projeto