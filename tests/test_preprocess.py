import pandas as pd
import pytest
from src.preprocess import preprocessar_geral

# Teste 1: Testa se as colunas esperadas estão presentes após o pré-processamento
def test_colunas_esperadas(tmp_path):
    test_file = tmp_path / "teste.csv"
    df_teste = pd.DataFrame({
        'year': [2010],
        'state': ['Bahia'],
        'month': ['August'],
        'number': [123.0]
    })
    df_teste.to_csv(test_file, index=False)

    df = preprocessar_geral(test_file)

    assert 'regiao' in df.columns
    assert 'date' in df.columns
    assert 'month' in df.columns
    assert df.loc[0, 'regiao'] == 'Nordeste'
    assert df['month'].iloc[0] == 8
    assert df['date'].iloc[0].year == 2010
    assert df['date'].iloc[0].month == 8

# Teste 2: Verifica os tipos de dados das colunas importantes
def test_tipos_de_dados(tmp_path):
    test_file = tmp_path / "tipos.csv"
    df_teste = pd.DataFrame({
        'year': [2012],
        'state': ['São Paulo'],
        'month': ['December'],
        'number': [456.0]
    })
    df_teste.to_csv(test_file, index=False)

    df = preprocessar_geral(test_file)

    assert pd.api.types.is_integer_dtype(df['month'])
    assert pd.api.types.is_datetime64_any_dtype(df['date'])

# Teste 3: Verifica se valores faltantes não existem em colunas críticas
def test_sem_valores_nulos(tmp_path):
    test_file = tmp_path / "nulos.csv"
    df_teste = pd.DataFrame({
        'year': [2013],
        'state': ['Bahia'],
        'month': ['January'],
        'number': [78.0]
    })
    df_teste.to_csv(test_file, index=False)

    df = preprocessar_geral(test_file)

    assert df['regiao'].isnull().sum() == 0
    assert df['month'].isnull().sum() == 0
    assert df['date'].isnull().sum() == 0

# Teste 4: Testa se estados inválidos são tratados corretamente
def test_estado_invalido(tmp_path):
    test_file = tmp_path / "estado_invalido.csv"
    df_teste = pd.DataFrame({
        'year': [2015],
        'state': ['Planeta Marte'],  # Estado inválido
        'month': ['June'],
        'number': [0.0]
    })
    df_teste.to_csv(test_file, index=False)

    df = preprocessar_geral(test_file)

    assert df['regiao'].iloc[0] in ['Desconhecido', None, 'Outro']  # Depende do seu mapeamento

# Teste 5: Garante que o número de linhas da entrada é igual à saída
def test_numero_linhas_constante(tmp_path):
    test_file = tmp_path / "linhas.csv"
    df_teste = pd.DataFrame({
        'year': [2010, 2011],
        'state': ['Bahia', 'São Paulo'],
        'month': ['August', 'December'],
        'number': [123.0, 456.0]
    })
    df_teste.to_csv(test_file, index=False)

    df = preprocessar_geral(test_file)

    assert len(df) == 2
