#main.py
from preprocess import*
from eda import eda

def main():
    caminho_csv = "./data/amazonfire.csv" 
    df_processado = preprocessar_geral(caminho_csv)
    #print(df_processado.head())  

    # analise exploratoria
    eda(df_processado)

if __name__ == "__main__":
    main()
