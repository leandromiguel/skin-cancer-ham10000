FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Adicionado --timeout=1000 e --retries=5 para evitar quedas no download grande
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copia a pasta app e o modelo da raiz para dentro do container
COPY ./app ./app
COPY best_hybrid_model.pth .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]