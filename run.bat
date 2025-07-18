@echo off
chcp 65001 > nul
echo Executando análise exploratória (EDA)...
python src/eda.py

echo Treinando Dados (ML)...
python src/model_trainer.py

echo Iniciando o dashboard Streamlit...
python -m streamlit run src/app.py

pause
