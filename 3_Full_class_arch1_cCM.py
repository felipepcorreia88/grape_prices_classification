# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:06:00 2025

@author: felip
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
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
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

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

# 4. Treinamento e Avaliação
def train_and_evaluate_classification(X_train, y_train, X_test, y_test, model_name):
    print(f"Executando modelo: {model_name}")
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Cria DataFrame com os resultados (usando a variável 'test' globalmente definida)
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

# 5. Execução Principal
file_path = 'Italia3_corrigido_tratado.xlsx'
df = load_data(file_path)
if df is not None:
    # Define features explicativas (excluindo 'Preco' e 'target')
    exog_features = [col for col in df.columns if col not in ['Preco', 'target']]
    
    # Dicionários para armazenar métricas ao longo das iterações
    accuracy_data = {model: [] for model in MODELS.keys()}
    f1_score_data = {model: [] for model in MODELS.keys()}
    months_range = list(range(0, 36))
    resultados = []
    
    # Loop para diferentes períodos de teste
    for months in range(0, 36):
        test_start_date = df.index.max() - pd.DateOffset(months=months)
        train = df[df.index < test_start_date]
        test = df[df.index >= test_start_date]  # variável 'test' usada na função de avaliação
        
        scaler_X = MinMaxScaler().fit(train[exog_features])
        X_train = scaler_X.transform(train[exog_features])
        y_train = train['target'].values
        X_test = scaler_X.transform(test[exog_features])
        y_test = test['target'].values
        
        for model_name in MODELS.keys():
            metrics, df_resultados = train_and_evaluate_classification(X_train, y_train, X_test, y_test, model_name)
            resultados.append({
                'months': months,
                'model': model_name,
                'metrics': metrics,
                'df_resultados': df_resultados
            })
            accuracy_data[model_name].append(metrics['Accuracy'])
            f1_score_data[model_name].append(metrics['F1-Score'])
    
    # Plot 1: Accuracy vs Number of Test Months
    plt.figure(figsize=(10, 6))
    for model, accuracies in accuracy_data.items():
        plt.plot(months_range, accuracies, label=model)
    plt.xlabel('Number of Test Months')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Test Months for Each Model')
    plt.legend()
    plt.grid()
    plt.savefig('class_arch1_cCM_accuracy_months.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: F1-Score vs Number of Test Months
    plt.figure(figsize=(10, 6))
    for model, f1_scores in f1_score_data.items():
        plt.plot(months_range, f1_scores, label=model)
    plt.xlabel('Number of Test Months')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Number of Test Months for Each Model')
    plt.legend()
    plt.grid()
    plt.savefig('class_arch1_cCM_f1score_months.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Feature Importance in Relation to Target
    # Calculates absolute correlation between each feature and the target variable
    feature_importance = pd.Series(
        data=np.abs(df[exog_features].corrwith(df['target'])),
        index=exog_features
    ).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="mako")
    plt.title('Feature Importance (Absolute Correlation with Target)')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Features')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('class_arch1_cCM_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Heatmap of Feature Correlation with Target
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[exog_features + ['target']].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Heatmap of Feature Correlation with Target')
    plt.tight_layout()
    plt.savefig('class_arch1_cCM_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Salvar os resultados em um arquivo Excel com duas planilhas:
    with pd.ExcelWriter('class_arch1_cCM_results.xlsx') as writer:
        # Planilha 1: Métricas de cada execução
        df_metrics = pd.DataFrame([{
            'months': r['months'],
            'model': r['model'],
            'Accuracy': r['metrics']['Accuracy'],
            'Precision': r['metrics']['Precision'],
            'Recall': r['metrics']['Recall'],
            'F1-Score': r['metrics']['F1-Score']
        } for r in resultados])
        df_metrics.to_excel(writer, sheet_name='Métricas', index=False)
    
        # Planilha 2: Detalhes dos resultados
        list_df_resultados = []
        for r in resultados:
            temp_df = r['df_resultados'].copy()
            temp_df['months'] = r['months']
            temp_df['model'] = r['model']
            list_df_resultados.append(temp_df)
        if list_df_resultados:
            df_detalhado = pd.concat(list_df_resultados)
            df_detalhado.to_excel(writer, sheet_name='Detalhes', index=True)
