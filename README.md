# Skin Cancer Classification — HAM10000

Classificação multiclasse de lesões dermatológicas com análise de interpretabilidade e auditoria de viés.

**Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10.015 imagens dermoscópicas, 7 classes  
**Curso:** OxeTech Academy — Inteligência Artificial e Aprendizado de Máquina (IA/ML)
## Descrição do problema 

    O projeto trata da classificação multiclasse de imagens dermatoscópicas do conjunto HAM10000.
O objetivo é identificar automaticamente o tipo de lesão de pele entre as 7 classes presentes no dataset, ajudando na detecção precoce de câncer de pele e outras anomalias dermatológicas.

    - Entrada: imagens dermatoscópicas de lesões cutâneas.
    - Saída: rótulo da classe da lesão.
    - Classes típicas do HAM10000: nevus melanocítico, carcinoma basocelular, carcinoma de células escamosas, melanoma, dermatite actínica, entre outras.

## Estrutura do projeto
```
root
├── README.md — documentação principal do projeto.
├── environment.yml — definição do ambiente Conda e dependências do projeto.
├── api/
│   ├── Dockerfile — contêiner da API.
│   ├── requirements.txt — dependências Python da API.
│   ├── README.md — instruções e detalhes da API.
│   └── app/
│       └── main.py — aplicação/API principal.
├── data/
│   └── processed/
│       ├── class_weights.json
│       ├── metadata_labeled.csv
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
├── notebooks/
│   ├── 01_eda.ipynb — análise exploratória de dados.
│   ├── baseline.ipynb — experimento baseline.
│   └── efficient-swin-skin-classification.ipynb — Código final
└── reports/
    └── figures/ — gráficos e resultados visuais.
```
## Como reproduzir o ambiente
\```bash
mamba env create -f environment.yml
conda activate skin-cancer
\```

O projeto foi executado no ambiente do kaggle, onde está armazenada todos os dados da sessão. Tanto o dataset original, quanto os novas imagens gerada para o treinamento.

O processo pode ser acompanhado por meio de 
    https://www.kaggle.com/code/jonhlucasalves/efficient-swin-skin-classification
na sua versão mais recentes `codigo final`.
## Equipe
- Leandro Miguel
- Jonh Lucas
