@echo off

echo Executando análise exploratória (EDA)...
python src/main.py

echo Iniciando o dashboard Streamlit...
streamlit run src/app.py

pause
