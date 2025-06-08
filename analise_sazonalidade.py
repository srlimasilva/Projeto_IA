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

# --- Configuração (sem alterações) ---
CSV_FILE = 'dataset.csv'
COLUMNS_MAPPING = {
    'ID_Pedido': 'ID_Pedido', 'Data_Pedido': 'Data_Pedido', 'ID_Cliente': 'ID_Cliente',
    'Segmento': 'Segmento', 'Pais': 'Pais', 'Cidade': 'Cidade', 'Estado': 'Estado',
    'ID_Produto': 'ID_Produto', 'Categoria': 'Categoria', 'SubCategoria': 'SubCategoria',
    'Valor_Venda': 'Valor_Venda',
}
# --- Caminhos (sem alterações) ---
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
    df.rename(columns={k: v for k, v in COLUMNS_MAPPING.items() if k in df.columns}, inplace=True)
    df["Data_Pedido"] = pd.to_datetime(df["Data_Pedido"], format='%d/%m/%Y', errors='coerce')
    df["Valor_Venda"] = pd.to_numeric(df["Valor_Venda"], errors='coerce')
    df.dropna(subset=['Data_Pedido', 'Valor_Venda'], inplace=True)
    df.sort_values('Data_Pedido', inplace=True)
    if df.empty:
        print("\nERRO: Dataset vazio.")
        exit()
    df_ts = df.set_index('Data_Pedido')['Valor_Venda'].resample('MS').sum()
    assert isinstance(df_ts.index, pd.DatetimeIndex)
except Exception as e:
    print(f"Ocorreu um erro na etapa 1: {e}")
    exit()

# --- CORREÇÃO: ADICIONANDO DE VOLTA AS ETAPAS DE ANÁLISE DE SAZONALIDADE ---
print("\n--- 2. Análise da Série Temporal e Decomposição Sazonal ---")
try:
    if len(df_ts) >= 24:
        decomposition = seasonal_decompose(df_ts, model='multiplicative', period=12, extrapolate_trend=12)
        joblib.dump(decomposition.trend, DECOMPOSITION_SAVE_PATH_TREND)
        joblib.dump(decomposition.seasonal, DECOMPOSITION_SAVE_PATH_SEASONAL)
        joblib.dump(decomposition.resid, DECOMPOSITION_SAVE_PATH_RESIDUAL)
        print("Componentes de decomposição sazonal salvos.")
    else:
        print("AVISO: Série temporal curta, decomposição não será salva.")
except Exception as e:
    print(f"AVISO: Não foi possível realizar a decomposição sazonal. Erro: {e}")

print("\n--- 3. Análise da Sazonalidade Média Mensal ---")
df_ts_monthly_avg = df_ts.groupby(df_ts.index.month).mean()
month_labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
df_ts_monthly_avg = df_ts_monthly_avg.rename(dict(zip(range(1, 13), month_labels)))
joblib.dump(df_ts_monthly_avg, MONTHLY_SALES_SAVE_PATH)
print(f"Sazonalidade média mensal salva em: {MONTHLY_SALES_SAVE_PATH}")

print("\n--- 4. Salvando Dados para o Streamlit ---")
joblib.dump(df_ts, SALES_TS_SAVE_PATH)
joblib.dump(COLUMNS_MAPPING, COLUMNS_MAPPING_SAVE_PATH)
print("Arquivos de análise de sazonalidade e mapeamento salvos com sucesso.")
# --- FIM DA CORREÇÃO ---

# --- 5. Preparação dos Dados para os Modelos ---
print("\n--- 5. Preparação dos Dados para os Modelos ---")
df['Ano'] = df['Data_Pedido'].dt.year
df['Mes'] = df['Data_Pedido'].dt.month
df['Dia'] = df['Data_Pedido'].dt.day
df['DiaDaSemana'] = df['Data_Pedido'].dt.dayofweek
features_columns_cat = ['Segmento', 'Pais', 'Cidade', 'Estado', 'Categoria', 'SubCategoria']
features_columns_num = ['Ano', 'Mes', 'Dia', 'DiaDaSemana']
available_features_cat = [col for col in features_columns_cat if col in df.columns]
df_model = df[available_features_cat + features_columns_num + ['Valor_Venda']].copy()
for col in available_features_cat:
    if df_model[col].dtype == 'object':
        df_model[col] = df_model[col].fillna('Desconhecido')
df_model.dropna(inplace=True)
df_model_encoded = pd.get_dummies(df_model, columns=available_features_cat, dummy_na=False)
X = df_model_encoded.drop('Valor_Venda', axis=1).astype(np.float32)
y = df_model_encoded['Valor_Venda'].astype(np.float32)
joblib.dump(X.columns.tolist(), DT_FEATURES_SAVE_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
tscv = TimeSeriesSplit(n_splits=5)


print("\n--- 6. Treinamento da Árvore de Decisão ---")
param_grid_dt = {'max_depth': [10, 20], 'min_samples_leaf': [1, 5, 10], 'min_samples_split': [2, 10, 20]}
grid_search_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_dt, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_dt_regressor = grid_search_dt.best_estimator_
y_pred_dt = best_dt_regressor.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"R² (DT): {r2_dt:.3f} com parâmetros {grid_search_dt.best_params_}")

print("\n--- 7. Treinamento do Random Forest ---")
param_grid_rf = {'n_estimators': [10, 20], 'max_depth': [10, 20], 'min_samples_leaf': [1, 5, 10]}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_regressor = grid_search_rf.best_estimator_
y_pred_rf = best_rf_regressor.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R² (RF): {r2_rf:.3f} com parâmetros {grid_search_rf.best_params_}")

print("\n--- 8. Treinamento do XGBoost ---")
param_grid_xgb = {'n_estimators': [10, 20], 'max_depth': [5, 7, 10], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]}
xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
grid_search_xgb = GridSearchCV(xgb_estimator, param_grid_xgb, cv=tscv, scoring='r2', n_jobs=-1, verbose=0)
grid_search_xgb.fit(X_train, y_train)
best_xgb_regressor = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb_regressor.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"R² (XGB): {r2_xgb:.3f} com parâmetros {grid_search_xgb.best_params_}")
# --- FIM DA CORREÇÃO DE HIPERPARÂMETROS ---

# --- 9. Comparar e Salvar o Melhor Modelo (Cálculos de métricas adicionados) ---
print("\n--- 9. Comparando e Salvando o Melhor Modelo ---")
models_results = {
    "DecisionTreeRegressor": (r2_dt, best_dt_regressor, y_pred_dt),
    "RandomForestRegressor": (r2_rf, best_rf_regressor, y_pred_rf),
    "XGBoostRegressor": (r2_xgb, best_xgb_regressor, y_pred_xgb),
}
best_model_type = max(models_results, key=lambda k: models_results[k][0])
best_r2_score, best_model, y_pred_best = models_results[best_model_type]
mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
current_metrics = {'mse': mse, 'rmse': rmse, 'r2': best_r2_score}
print(f"\nO MELHOR MODELO GERAL É: {best_model_type} com R² de {best_r2_score:.4f}")
joblib.dump(best_model, BEST_MODEL_SAVE_PATH)
joblib.dump(best_model_type, MODEL_TYPE_SAVE_PATH)
joblib.dump(current_metrics, MODEL_METRICS_SAVE_PATH)
print("Melhor modelo e suas métricas foram salvos.")

# --- 10. Visualização da Árvore (sem alterações significativas) ---
if best_model_type == "DecisionTreeRegressor":
    print("\n--- 10. Gerando Imagem da Árvore de Decisão ---")
    plt.figure(figsize=(25, 12))
    viz_depth = min(best_model.get_params().get('max_depth', 5), 5)
    plot_tree(best_model, filled=True, feature_names=X.columns.tolist(), fontsize=9, max_depth=viz_depth)
    plt.title(f'Árvore de Decisão (Visualização Limitada a Profundidade {viz_depth})')
    plt.savefig(TREE_IMAGE_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Imagem da Árvore de Decisão salva.")
else:
    if os.path.exists(TREE_IMAGE_PATH):
        os.remove(TREE_IMAGE_PATH)

print("\nAnálise e Modelagem Concluída!")