#eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df: pd.DataFrame, pasta_plots="./output/plots"):
    """
    Realiza Análise Exploratória de Dados (EDA) sobre dados de incêndios florestais no Brasil.
    Gera gráficos e os salva na pasta especificada.

    Parâmetros:
        df (pd.DataFrame): DataFrame com colunas ['year', 'month_num', 'state', 'number', 'regiao']
        pasta_plots (str): Caminho para salvar os gráficos gerados
    """
    # Criar pasta para os plots
    os.makedirs(pasta_plots, exist_ok=True)

    # Checagem básica
    colunas_esperadas = {'year', 'month_num', 'state', 'number', 'regiao'}
    if not colunas_esperadas.issubset(df.columns):
        raise ValueError(f"O DataFrame deve conter as colunas: {colunas_esperadas}")

    # ---------- 4.1 Total de incêndios por ano ----------
    df_total_ano = df.groupby("year", as_index=False)["number"].sum()
    df_total_ano["year"] = df_total_ano["year"].astype(int)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_total_ano, x='year', y='number', marker='o')
    plt.title("Número Total de Incêndios por Ano")
    plt.xlabel("Ano")
    plt.ylabel("Número de Incêndios")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/incendios_por_ano.png")
    plt.close()

    # ---------- 4.2 Distribuição mensal dos incêndios ----------
    somas = df.groupby('month_num')['number'].sum().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=somas, x='month_num', y='number', color='skyblue', edgecolor='black')
    plt.title("Número total de queimadas por mês")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade de queimadas")
    plt.xticks(range(0, 12), labels=range(1, 13))
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/incendios_por_mes.png")
    
    # ---------- 4.3 Total de incêndios por estado ----------
    total_estado = df.groupby('state')['number'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=total_estado.values, y=total_estado.index, palette='Reds_r', hue=total_estado.index, legend=False)
    plt.title("Total de Incêndios por Estado (1998–2017)")
    plt.xlabel("Número de Incêndios")
    plt.ylabel("Estado")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/total_incendios_estado.png")
    plt.close()

    # ---------- 4.5 Tendência mensal por estado e região ----------
    df_media = df.groupby(['regiao', 'state', 'month_num'])['number'].mean().reset_index()

    for regiao in df_media['regiao'].unique():
        df_regiao = df_media[df_media['regiao'] == regiao]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_regiao, x='month_num', y='number', hue='state', marker='o')
        plt.title(f"Tendência Mensal de Incêndios - Região {regiao}")
        plt.xlabel("Mês")
        plt.ylabel("Média de Incêndios")
        plt.legend(title='Estado', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{pasta_plots}/tendencia_mensal_{regiao}.png")
        plt.close()

    # ---------- 4.6 Regressão mês x incêndios ----------
    # Média de incêndios por mês (agregado para todos os anos)
    media_mes = df.groupby('month_num')['number'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=media_mes, x='month_num', y='number', marker='o', color='b')
    plt.title("Média mensal do número de incêndios (todos os anos)")
    plt.xlabel("Mês")
    plt.ylabel("Número médio de incêndios")
    plt.xticks(ticks=range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/media_mensal_incendios.png")
    plt.close()

    # Boxplot mostrando a distribuição dos incêndios por mês
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='number', marker='o', color='green')
    plt.title("Tendência anual do número de incêndios")
    plt.xlabel("Ano")
    plt.ylabel("Número de incêndios")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/tendencia_anual_incendios.png")
    plt.close()


    # ---------- 4.8 Heatmap: incêndios médios por estado e mês ----------
    pivot = df.pivot_table(index='state', columns='month_num', values='number', aggfunc='mean')
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap='Reds', cbar_kws={'label': 'Incêndios Médios'})
    plt.title("Heatmap: Incêndios Médios por Estado e Mês")
    plt.xlabel("Mês")
    plt.ylabel("Estado")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/heatmap_estado_mes.png")
    plt.close()

    print(f"[OK] Análise exploratória concluída. Gráficos salvos em: '{pasta_plots}'")
