# utils.py - Módulo de Funções Auxiliares para o Pipeline de Séries Temporais

import os
import pandas as pd
import numpy as np
import random
from statsmodels.datasets import get_rdataset
from sklearn.metrics import mean_squared_error

# =========================================================
# FUNÇÕES DE SETUP E PROCESSAMENTO DE DADOS
# =========================================================

def definir_seed(seed_value=42):
    """Define a semente para garantir a reprodutibilidade."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def carregar_serie(nome, pasta_cache="./data/bronze"):
    """
    Carrega uma série temporal de forma eficiente, utilizando um cache local.
    """
    nome_base = nome.lower()
    os.makedirs(pasta_cache, exist_ok=True)
    caminho_arquivo = os.path.join(pasta_cache, f"{nome_base}.csv")
    
    if os.path.exists(caminho_arquivo):
        #print(f"Carregando dataset '{nome}' do cache local: {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo, parse_dates=['date'], index_col='date')
        df['value'].name = nome
        return df['value']

    print(f"Cache não encontrado. Buscando dados de '{nome}' via statsmodels...")
    
    serie = None
    if nome_base == "airpassengers":
        dados = get_rdataset("AirPassengers", package="datasets").data
        serie = pd.Series(dados['value'].values, index=pd.date_range(start="1949-01-01", periods=len(dados), freq="MS"), name=nome)
    elif nome_base == "lynx":
        dados = get_rdataset("lynx", package="datasets").data
        serie = pd.Series(dados['value'].values, index=pd.date_range(start="1821", periods=len(dados), freq="YE-DEC"), name=nome)
    elif nome_base == "co2":
        dados = get_rdataset("CO2", package="datasets").data
        dados = dados.ffill()
        serie = pd.Series(dados['value'].values, index=pd.date_range(start="1958-03-29", periods=len(dados), freq="MS"), name=nome)
    elif nome_base == "austres":
        dados = get_rdataset("austres", package="datasets").data
        serie = pd.Series(dados['value'].values, index=pd.date_range(start="1971-03-01", periods=len(dados), freq="QS-MAR"), name=nome)
    elif nome_base == "nottem":
        dados = get_rdataset("nottem", package="datasets").data
        serie = pd.Series(dados['value'].values, index=pd.date_range(start="1920-01-01", periods=len(dados), freq="MS"), name=nome)
    else:
        raise ValueError(f"Lógica de download para a série '{nome}' não implementada.")

    if serie is not None:
        print(f"-> Salvando cópia do dataset '{nome}' em cache: {caminho_arquivo}")
        df_para_salvar = pd.DataFrame({"date": serie.index, "value": serie.values})
        df_para_salvar.to_csv(caminho_arquivo, index=False)
    
    return serie

def dividir_serie_temporal(serie, percentual_treino=0.85):
    """Divide a série em conjuntos de treino e teste."""
    tamanho_total = len(serie)
    ponto_corte_treino = int(tamanho_total * percentual_treino)
    treino = serie.iloc[:ponto_corte_treino]
    teste = serie.iloc[ponto_corte_treino:]
    return treino, teste

def preparar_dados_para_neuralforecast(serie, nome_serie):
    """Formata um DataFrame para o padrão exigido pela biblioteca NeuralForecast."""
    df = serie.reset_index()
    df.columns = ['ds', 'y']
    df['unique_id'] = nome_serie
    return df

# =========================================================
# FUNÇÃO DE CÁLCULO DE MÉTRICAS
# =========================================================

def calcular_metricas(y_true, y_pred, y_train):
    """Calcula um dicionário de métricas de avaliação: RMSE, MAPE e MASE."""
    # Garante que os inputs sejam arrays numpy para os cálculos
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Evita divisão por zero no MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    
    # Cálculo do MASE
    n = len(y_train)
    # Erro médio absoluto da previsão ingênua (naive forecast) no conjunto de treino
    d = np.sum(np.abs(np.diff(y_train))) / (n - 1) if n > 1 else np.nan
    
    # Erro absoluto médio no conjunto de teste
    mae_test = np.mean(np.abs(y_true - y_pred))
    
    mase = mae_test / d if d is not np.nan and d > 0 else np.inf
    
    return {'RMSE': rmse, 'MAPE(%)': mape, 'MASE': mase}