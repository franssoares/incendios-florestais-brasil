#src/preprocess.py
import pandas as pd
import os # Importar para usar os.path.join e os.path.dirname

def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.

    Args:
        caminho (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: O DataFrame carregado.
    """
    df = pd.read_csv(caminho, encoding="utf-8").copy()
    return df

def salvar_copia(df: pd.DataFrame):
    """
    Salva uma cópia do DataFrame processado em um novo arquivo CSV.
    O caminho é ajustado para a pasta 'data' na raiz do projeto,
    garantindo que o diretório exista.

    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
    """
    # Obtém o diretório do arquivo atual (preprocess.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Volta um nível para a raiz do projeto (de 'src' para 'analise_incendios_brasil')
    project_root_dir = os.path.dirname(current_dir)
    # Constrói o caminho completo para a pasta 'data' dentro da raiz do projeto
    data_dir_path = os.path.join(project_root_dir, 'data')

    # Garante que o diretório 'data' exista antes de tentar salvar o arquivo
    os.makedirs(data_dir_path, exist_ok=True)

    # Constrói o caminho completo para o arquivo de saída
    caminho_salvar = os.path.join(data_dir_path, 'amazonfire_edit.csv')
    df.to_csv(caminho_salvar, index=False, encoding='utf-8')

def remover_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com valores nulos na coluna 'number'.

    Args:
        df (pd.DataFrame): O DataFrame original.

    Returns:
        pd.DataFrame: O DataFrame sem valores nulos na coluna 'number'.
    """
    return df[df["number"].notnull()]

def remover_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers da coluna 'number' usando o método IQR.
    (Atualmente, esta função está comentada na pipeline principal 'preprocessar_geral')

    Args:
        df (pd.DataFrame): O DataFrame original.

    Returns:
        pd.DataFrame: O DataFrame com outliers removidos.
    """
    q1 = df["number"].quantile(0.25)
    q3 = df["number"].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr

    # Cria filtro booleano
    filtro = (df["number"] >= lim_inf) & (df["number"] <= lim_sup)

    # Filtra dados e converte para int
    df_filtrado = df[filtro].copy()
    df_filtrado["number"] = df_filtrado["number"].astype(int)

    return df_filtrado

def mapear_meses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia os nomes dos meses em português para números e aliases.

    Args:
        df (pd.DataFrame): O DataFrame original com a coluna 'month'.

    Returns:
        pd.DataFrame: O DataFrame com as novas colunas 'month_num' e 'month_alias'.
    """
    meses_info = {
        'janeiro':  {'num': 1,  'alias': 'jan'},
        'fevereiro':{'num': 2,  'alias': 'fev'},
        'março':    {'num': 3,  'alias': 'mar'},
        'abril':    {'num': 4,  'alias': 'abr'},
        'maio':     {'num': 5,  'alias': 'mai'},
        'junho':    {'num': 6,  'alias': 'jun'},
        'julho':    {'num': 7,  'alias': 'jul'},
        'agosto':   {'num': 8,  'alias': 'ago'},
        'setembro': {'num': 9,  'alias': 'set'},
        'outubro':  {'num': 10, 'alias': 'out'},
        'novembro': {'num': 11, 'alias': 'nov'},
        'dezembro': {'num': 12, 'alias': 'dez'}
    }

    # Padroniza o texto
    meses_clean = df['month'].astype(str).str.strip().str.lower()

    # Cria colunas vazias
    df['month_num'] = meses_clean.map(lambda x: meses_info.get(x, {}).get('num'))
    df['month_alias'] = meses_clean.map(lambda x: meses_info.get(x, {}).get('alias'))

    return df

def criar_coluna_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma coluna 'date' no formato datetime a partir do ano e número do mês.

    Args:
        df (pd.DataFrame): O DataFrame original com as colunas 'year' e 'month_num'.

    Returns:
        pd.DataFrame: O DataFrame com a nova coluna 'date'.
    """
    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month_num'], day=1))
    return df

def adicionar_regiao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma coluna 'regiao' ao DataFrame com base no estado.

    Args:
        df (pd.DataFrame): O DataFrame original com a coluna 'state'.

    Returns:
        pd.DataFrame: O DataFrame com a nova coluna 'regiao'.
    """
    regioes = {
        "Norte": ["Acre", "Amapa", "Amazonas", "Pará", "Rondonia", "Roraima", "Tocantins"],
        "Nordeste": ["Alagoas", "Bahia", "Ceara", "Maranhao", "Paraiba", "Pernambuco", "Piauí", "Rio Grande do Norte", "Sergipe"],
        "Centro-Oeste": ["Goias", "Mato Grosso", "Mato Grosso do Sul", "Distrito Federal"],
        "Sudeste": ["Espirito Santo", "Minas Gerais", "Rio de Janeiro", "Sao Paulo"],
        "Sul": ["Paraná", "Rio Grande do Sul", "Santa Catarina"]
    }

    # Cria dicionário plano {estado: regiao}
    estado_para_regiao = {estado: regiao for regiao, estados in regioes.items() for estado in estados}

    # Faz o mapeamento direto
    df['regiao'] = df['state'].map(estado_para_regiao)

    return df

def preprocessar_geral(caminho_csv: str) -> pd.DataFrame:
    """
    Função principal para carregar e pré-processar o DataFrame.

    Args:
        caminho_csv (str): O caminho para o arquivo CSV de entrada.

    Returns:
        pd.DataFrame: O DataFrame processado.
    """
    df = carregar_dados(caminho_csv)
    df = remover_nulos(df)
    # A função remover_outliers está comentada, você pode ativá-la se desejar
    # df = remover_outliers(df)
    df = mapear_meses(df)
    df = criar_coluna_data(df)
    df = adicionar_regiao(df)
    salvar_copia(df) # Salva uma cópia do DataFrame processado
    return df
