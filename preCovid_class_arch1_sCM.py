# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:06:00 2025

@author: felip
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Engenharia de Features Temporais
def create_time_features(df, target_col, lags=7, window=14):
    """
    Cria features temporais como lags, média móvel e desvio padrão.
    """
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df['rolling_mean'] = df[target_col].rolling(window=window).mean()
    df['rolling_std'] = df[target_col].rolling(window=window).std()
    return df.dropna()

# 2. Carregamento e Preparação dos Dados
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, parse_dates=['Data'], index_col='Data').sort_index()
        if 'Preco' not in df.columns:
            raise ValueError("A coluna 'Preco' não foi encontrada no DataFrame.")
        df = create_time_features(df, 'Preco')
        df['target'] = (df['Preco'].diff() > 0).astype(int)
        return df.dropna()
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

# 3. Modelos de Classificação
MODELS = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100),
}

def create_mlp(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Treinamento e Avaliação
def train_and_evaluate_classification(X_train, y_train, X_test, y_test, model_name):
    print(f"Executando modelo: {model_name}")
    if model_name == "MLP":
        model = create_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        model = MODELS[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    df_resultados = pd.DataFrame({
        'Preco_anterior': test['lag_1'].values,
        'Preco_atual': test['Preco'].values,
        'Real': test['target'].values,
        'Previsao': y_pred
    }, index=test.index)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    print(f"Resultados para {model_name}: {metrics}")
    return metrics, df_resultados

# 5. Execução
file_path = 'Italia3_nao_corrigido_tratado_preCovid.xlsx'
df = load_data(file_path)
if df is not None:
    exog_features = [col for col in df.columns if col not in ['Preco', 'target']]
    accuracy_data = {model: [] for model in MODELS.keys()}
    months_range = list(range(0, 36))
    resultados = []
    for months in range(0, 36):
        test_start_date = df.index.max() - pd.DateOffset(months=months)
        train = df[df.index < test_start_date]
        test = df[df.index >= test_start_date]
        scaler_X = MinMaxScaler().fit(train[exog_features])
        X_train = scaler_X.transform(train[exog_features])
        y_train = train['target'].values
        X_test = scaler_X.transform(test[exog_features])
        y_test = test['target'].values
        for model_name in MODELS.keys():
            metrics, df_resultados = train_and_evaluate_classification(X_train, y_train, X_test, y_test, model_name)
            resultados.append({'months': months, 'model': model_name, 'metrics': metrics, 'df_resultados': df_resultados})
            accuracy_data[model_name].append(metrics['Accuracy'])
    
    # Gerar gráfico de Acurácia vs Número de Meses para cada modelo
    plt.figure(figsize=(10, 6))
    for model, accuracies in accuracy_data.items():
        plt.plot(months_range, accuracies, label=model)
    plt.xlabel('Número de Meses de Teste')
    plt.ylabel('Acurácia')
    plt.title('Acurácia vs Número de Meses para cada Modelo')
    plt.legend()
    plt.grid()
    plt.show()