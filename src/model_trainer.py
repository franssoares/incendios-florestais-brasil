# model_trainer.py
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def prepare_data(df):
    """Prepara os dados para o modelo de regressão."""
    # Agrupar por ano e mês para ter dados temporais consistentes
    df_agg = df.groupby(['year', 'month_num'])['number'].sum().reset_index()
    
    # Criar features (podemos adicionar mais features temporais se necessário)
    X = df_agg[['year', 'month_num']]
    y = df_agg['number']
    
    return X, y

def train_linear_regression(X, y):
    """Treina o modelo de regressão linear."""
    # Dividir dados (usando shuffle=False para manter ordem temporal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained - MSE: {mse:.2f}, R2: {r2:.2f}")
    
    return model, X_test, y_test, y_pred

def save_model(model, filename='linear_regression_model.pkl'):
    """Serializa o modelo usando pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def save_results(X_test, y_test, y_pred, filename='model_results.csv'):
    """Salva os resultados do modelo em CSV."""
    results = pd.DataFrame({
        'year': X_test['year'],
        'month': X_test['month_num'],
        'actual': y_test,
        'predicted': y_pred
    })
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def train_and_save_model(csv_path):
    """Função principal para treinar e salvar o modelo."""
    # Carregar e pré-processar dados
    df = pd.read_csv(csv_path)
    X, y = prepare_data(df)
    
    # Treinar modelo
    model, X_test, y_test, y_pred = train_linear_regression(X, y)
    
    # Salvar artefatos
    save_model(model)
    save_results(X_test, y_test, y_pred)
    
    return model

if __name__ == "__main__":
    # Supondo que o arquivo de dados está na pasta data
    csv_path = os.path.join('data', 'amazonfire_edit.csv')
    train_and_save_model(csv_path)