import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import os
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb

# --- Configuração ---
CSV_FILE = 'dataset.csv'

# Mapeamento dos nomes das colunas (sem alterações)
COLUMNS_MAPPING = {
    'ID_Pedido': 'ID_Pedido', 'Data_Pedido': 'Data_Pedido', 'ID_Cliente': 'ID_Cliente',
    'Segmento': 'Segmento', 'Pais': 'Pais', 'Cidade': 'Cidade', 'Estado': 'Estado',
    'ID_Produto': 'ID_Produto', 'Categoria': 'Categoria', 'SubCategoria': 'SubCategoria',
    'Valor_Venda': 'Valor_Venda',
}

# Caminhos para salvar os resultados (sem alterações)
SALES_TS_SAVE_PATH = 'sales_time_series.joblib'
DECOMPOSITION_SAVE_PATH_TREND = 'ts_trend.joblib'
DECOMPOSITION_SAVE_PATH_SEASONAL = 'ts_seasonal.joblib'
DECOMPOSITION_SAVE_PATH_RESIDUAL = 'ts_residual.joblib'
MONTHLY_SALES_SAVE_PATH = 'monthly_sales_summary.joblib'
COLUMNS_MAPPING_SAVE_PATH = 'columns_mapping.joblib'
BEST_MODEL_SAVE_PATH = 'best_sales_prediction_model.joblib'
MODEL_TYPE_SAVE_PATH = 'model_type.joblib'
DT_FEATURES_SAVE_PATH = 'decision_tree_features.joblib'
TREE_IMAGE_PATH = 'decision_tree_plot_tree.png'
MODEL_METRICS_SAVE_PATH = 'model_metrics.joblib'

print("Iniciando a Análise e Modelagem com Otimizações...\n")

# --- 1. Carregamento e Pré-processamento de Dados ---
print("--- 1. Carregamento e Pré-processamento de Dados ---")
try:
    df = pd.read_csv(CSV_FILE, sep=',', header=0)
    print(f"Dataset '{CSV_FILE}' carregado com sucesso.")
    df.rename(columns={k: v for k, v in COLUMNS_MAPPING.items() if k in df.columns}, inplace=True)
    print("\nNomes das colunas renomeados.")

    required_cols_for_analysis = ['Data_Pedido', 'Valor_Venda']
    if not all(col in df.columns for col in required_cols_for_analysis):
        print(f"\nERRO: Colunas essenciais ({', '.join(required_cols_for_analysis)}) não encontradas.")
        exit()

    df["Data_Pedido"] = pd.to_datetime(df["Data_Pedido"], format='%d/%m/%Y', errors='coerce')
    df["Valor_Venda"] = pd.to_numeric(df["Valor_Venda"], errors='coerce')

    df.dropna(subset=['Data_Pedido', 'Valor_Venda'], inplace=True)
    
    # --- MODIFICAÇÃO --- -> Ordenar os dados pela data para a validação cruzada
    df.sort_values('Data_Pedido', inplace=True)
    print("\nDataFrame ordenado por 'Data_Pedido' para garantir a sequência temporal.")

    if df.empty:
        print("\nERRO: Dataset vazio após pré-processamento.")
        exit()

    df_ts = df.set_index('Data_Pedido')['Valor_Venda'].resample('MS').sum()

except FileNotFoundError:
    print(f"Erro: O arquivo '{CSV_FILE}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar ou processar o dataset: {e}")
    exit()

# As seções 2, 3 e 4 (Análise de Sazonalidade e salvamento) permanecem as mesmas.
# Para economizar espaço, vamos pular a exibição do código delas.
# O código original para essas seções funciona perfeitamente.

# --- 5. Preparação dos Dados para os Modelos ---
print("\n--- 5. Preparação dos Dados para os Modelos ---")

# --- NOVA ETAPA: Engenharia de Features a partir da Data ---
print("\nCriando novas features a partir da data (Ano, Mês, Dia, Dia da Semana)...")
df['Ano'] = df['Data_Pedido'].dt.year
df['Mes'] = df['Data_Pedido'].dt.month
df['Dia'] = df['Data_Pedido'].dt.day
df['DiaDaSemana'] = df['Data_Pedido'].dt.dayofweek # Segunda=0, Domingo=6

# --- MODIFICAÇÃO --- -> Adicionar as novas features numéricas à lista
features_columns_cat = ['Segmento', 'Pais', 'Cidade', 'Estado', 'Categoria', 'SubCategoria']
features_columns_num = ['Ano', 'Mes', 'Dia', 'DiaDaSemana']

available_features_cat = [col for col in features_columns_cat if col in df.columns]
df_model = df[available_features_cat + features_columns_num + ['Valor_Venda']].copy()

for col in available_features_cat:
    if df_model[col].dtype == 'object':
        df_model[col].fillna('Desconhecido', inplace=True)

df_model.dropna(inplace=True)

if df_model.empty:
    print("\nERRO: Dataset para modelagem vazio.")
    exit()

print("\nConvertendo variáveis categóricas para numéricas (One-Hot Encoding)...")
df_model_encoded = pd.get_dummies(df_model, columns=available_features_cat, dummy_na=False)

X = df_model_encoded.drop('Valor_Venda', axis=1).astype(np.float32)
y = df_model_encoded['Valor_Venda'].astype(np.float32)

joblib.dump(X.columns.tolist(), DT_FEATURES_SAVE_PATH)
print(f"Novos nomes das features dos modelos salvos em: {DT_FEATURES_SAVE_PATH}")

# --- MODIFICAÇÃO --- -> A divisão treino/teste agora não precisa embaralhar (shuffle=False)
# pois a validação cruzada cuidará da sequência temporal.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
print(f"Tamanho do conjunto de teste: {len(X_test)} amostras")

# --- NOVA ETAPA: Definir a estratégia de Validação Cruzada para Séries Temporais ---
print("\nDefinindo a estratégia de validação cruzada com TimeSeriesSplit.")
tscv = TimeSeriesSplit(n_splits=5)

# --- 6. Treinamento e Otimização da Árvore de Decisão ---
print("\n--- 6. Treinamento e Otimização da Árvore de Decisão ---")
param_grid_dt = {'max_depth': [5, 10, 25], 'min_samples_leaf': [1, 5], 'min_samples_split': [2, 5]}
grid_search_dt = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                              param_grid=param_grid_dt,
                              cv=tscv, # --- MODIFICAÇÃO ---
                              scoring='r2', n_jobs=-1, verbose=0)
grid_search_dt.fit(X_train, y_train)
best_dt_regressor = grid_search_dt.best_estimator_
y_pred_dt = best_dt_regressor.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"R² (DT) no conjunto de teste: {r2_dt:.4f}")

# --- 7. Treinamento e Otimização do Random Forest ---
print("\n--- 7. Treinamento e Otimização do Random Forest ---")
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_leaf': [1, 5]}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=param_grid_rf,
                              cv=tscv, # --- MODIFICAÇÃO ---
                              scoring='r2', n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train)
best_rf_regressor = grid_search_rf.best_estimator_
y_pred_rf = best_rf_regressor.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R² (RF) no conjunto de teste: {r2_rf:.4f}")

# --- 8. Treinamento e Otimização do XGBoost com GPU ---
print("\n--- 8. Treinamento e Otimização do XGBoost (com GPU) ---")

# --- MODIFICAÇÃO --- -> Aumentar a grade de busca para explorar mais opções
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'gamma': [0, 0.1] # Parâmetro para controle de overfitting
}

# --- MODIFICAÇÃO --- -> Adicionar 'tree_method' para usar a GPU
# !! IMPORTANTE !! -> Isso só funcionará se você tiver uma GPU NVIDIA e o CUDA Toolkit instalado!
# Se der erro, mude para 'tree_method':'hist' ou remova o parâmetro para usar a CPU.
xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror',
                                 eval_metric='rmse',
                                 random_state=42,
                                 n_jobs=-1,
                                 tree_method='hist') # <-- A MÁGICA ACONTECE AQUI!

grid_search_xgb = GridSearchCV(estimator=xgb_estimator,
                               param_grid=param_grid_xgb,
                               cv=tscv, # --- MODIFICAÇÃO ---
                               scoring='r2',
                               n_jobs=-1, # n_jobs do GridSearchCV continua -1
                               verbose=1) # Aumentar verbose para ver o progresso

print("Iniciando a busca de hiperparâmetros para o XGBoost...")
grid_search_xgb.fit(X_train, y_train)

best_xgb_regressor = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb_regressor.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"\n--- Resultados do XGBoost Otimizado (GPU) ---")
print(f"Melhores hiperparâmetros (XGB): {grid_search_xgb.best_params_}")
print(f"R² (XGB) no conjunto de teste: {r2_xgb:.4f}")


# --- 9. Comparar Modelos e Salvar o Melhor ---
# (O código desta seção para comparar e salvar o melhor modelo já está correto e não precisa de alterações)
print("\n--- 9. Comparando e Salvando o Melhor Modelo ---")
# Coletando métricas para salvar
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

best_model = None
model_type = ""
best_r2_score = -float('inf')
current_metrics = {}

if r2_dt > best_r2_score:
    best_r2_score, best_model, model_type = r2_dt, best_dt_regressor, "DecisionTreeRegressor"
    current_metrics = {'mse': mse_dt, 'rmse': rmse_dt, 'r2': r2_dt}

if r2_rf > best_r2_score:
    best_r2_score, best_model, model_type = r2_rf, best_rf_regressor, "RandomForestRegressor"
    current_metrics = {'mse': mse_rf, 'rmse': rmse_rf, 'r2': r2_rf}

if r2_xgb > best_r2_score:
    best_r2_score, best_model, model_type = r2_xgb, best_xgb_regressor, "XGBoostRegressor"
    current_metrics = {'mse': mse_xgb, 'rmse': rmse_xgb, 'r2': r2_xgb}

print(f"\nO MELHOR MODELO GERAL É: {model_type} com R² de {best_r2_score:.4f}")

joblib.dump(best_model, BEST_MODEL_SAVE_PATH)
joblib.dump(model_type, MODEL_TYPE_SAVE_PATH)
joblib.dump(current_metrics, MODEL_METRICS_SAVE_PATH)
print(f"Melhor modelo ({model_type}) e suas métricas foram salvos.")
# --- 10. Visualização e Salvamento da Árvore de Decisão (apenas se for o melhor modelo) ---
# Gerar a imagem da árvore SOMENTE se o melhor modelo for uma Decision Tree
if model_type == "DecisionTreeRegressor":
    print("\n--- 10. Gerando e Salvando Imagem da Árvore de Decisão ---")
    plt.figure(figsize=(25, 12))
    # Usar a profundidade ideal encontrada para o melhor DecisionTreeRegressor para a visualização,
    # mas limitar para 3-5 para legibilidade se a ideal for muito grande
    max_depth_dt = best_dt_regressor.get_params().get('max_depth', None)
    viz_depth = min(max_depth_dt if max_depth_dt is not None else 5, 5) # Limita a profundidade para visualização
    plot_tree(best_dt_regressor, filled=True, feature_names=X.columns.tolist(),
              fontsize=9, max_depth=viz_depth)
    plt.title(f'Árvore de Decisão (Profundidade Máxima para Visualização: {viz_depth})')
    plt.savefig(TREE_IMAGE_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Imagem da Árvore de Decisão salva em: {TREE_IMAGE_PATH}")
else:
    print("\n--- 10. Não Gerando Imagem da Árvore de Decisão ---")
    print("O melhor modelo é um Random Forest ou XGBoost, que não possui uma única árvore para visualização direta.")
    # Opcional: Remover arquivo de imagem antigo se existir
    if os.path.exists(TREE_IMAGE_PATH):
        os.remove(TREE_IMAGE_PATH)
        print(f"Removido arquivo de imagem de árvore anterior: {TREE_IMAGE_PATH}")


print("\nAnálise e Modelagem Concluída!")