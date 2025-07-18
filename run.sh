#!/bin/bash

echo "Executando análise exploratória (EDA)..."
python3 src/main.py

echo "Iniciando o dashboard Streamlit..."
streamlit run src/app.py
