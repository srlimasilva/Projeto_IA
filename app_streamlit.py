import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.tree import plot_tree # Necessário para a função plot_tree, embora não seja usada diretamente para o dashboard

# --- Caminhos dos arquivos da análise ---
SALES_TS_SAVE_PATH = 'sales_time_series.joblib'
DECOMPOSITION_SAVE_PATH_TREND = 'ts_trend.joblib'
DECOMPOSITION_SAVE_PATH_SEASONAL = 'ts_seasonal.joblib'
DECOMPOSITION_SAVE_PATH_RESIDUAL = 'ts_residual.joblib'
MONTHLY_SALES_SAVE_PATH = 'monthly_sales_summary.joblib'
COLUMNS_MAPPING_SAVE_PATH = 'columns_mapping.joblib'

# Caminhos para o MELHOR modelo de regressão
BEST_MODEL_SAVE_PATH = 'best_sales_prediction_model.joblib'
MODEL_TYPE_SAVE_PATH = 'model_type.joblib' # Para saber se é DT, RF ou XGB
DT_FEATURES_SAVE_PATH = 'decision_tree_features.joblib' # Nomes das features
TREE_IMAGE_PATH = 'decision_tree_plot_tree.png' # Caminho para a imagem da árvore (se o melhor for DT)
MODEL_METRICS_SAVE_PATH = 'model_metrics.joblib' # Caminho para carregar as métricas do modelo

# --- Título da Aplicação ---
st.set_page_config(
    page_title="Análise de Sazonalidade e Previsão de Vendas",
    layout="wide",
    initial_sidebar_state="auto"
)
st.title("📈 Análise de Sazonalidade das Vendas e Previsão de Vendas")
st.markdown("Este aplicativo analisa padrões sazonais em dados de vendas e permite interagir com o modelo de previsão mais eficaz.")

# --- Funções Auxiliares para Análise de Sazonalidade ---

@st.cache_data
def perform_seasonality_analysis(df: pd.DataFrame, columns_mapping: dict, data_col_name: str, sales_col_name: str):
    df_processed = df.copy()

    # Tenta renomear se as colunas já não estiverem no formato mapeado
    if not all(col in df_processed.columns for col in [data_col_name, sales_col_name]):
        # Apenas as chaves que realmente existem em df_processed.columns serão renomeadas
        df_processed.rename(columns={k: v for k, v in columns_mapping.items() if k in df_processed.columns}, inplace=True)
        if not all(col in df_processed.columns for col in [data_col_name, sales_col_name]):
            st.error(f"Erro: As colunas essenciais '{data_col_name}' (Data do Pedido) ou '{sales_col_name}' (Valor de Venda) não foram encontradas no dataset após a tentativa de renomear.")
            return None, None, None, None, None

    # Tratamento de nulos antes de converter tipos
    df_processed.dropna(subset=[data_col_name, sales_col_name], inplace=True)
    if df_processed.empty:
        st.warning("O dataset está vazio após remover linhas com valores ausentes nas colunas essenciais para a análise de sazonalidade.")
        return None, None, None, None, None

    # Conversão de tipos: Data_Pedido no formato 'DD/MM/YYYY' e Valor_Venda numérico
    df_processed[data_col_name] = pd.to_datetime(df_processed[data_col_name], format='%d/%m/%Y', errors='coerce')
    df_processed[sales_col_name] = pd.to_numeric(df_processed[sales_col_name], errors='coerce')
    df_processed.dropna(subset=[data_col_name, sales_col_name], inplace=True) # Remover nulos após coerção
    
    if df_processed.empty:
        st.warning("O dataset está vazio após a conversão de tipos de dados.")
        return None, None, None, None, None

    sales_ts = df_processed.set_index(data_col_name)[sales_col_name].resample('MS').sum()
    monthly_sales_avg = sales_ts.groupby(sales_ts.index.month).mean() # type: ignore
    # No analise_e_modelagem.py, o monthly_sales_avg.index é definido com nomes de mês.
    
    trend, seasonal, residual = None, None, None
    try:
        if len(sales_ts) >= 2 * 12: # Pelo menos 2 ciclos anuais completos
            decomposition = seasonal_decompose(sales_ts, model='multiplicative', period=12, extrapolate_trend='freq') # type: ignore
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
        else:
            st.warning("Série temporal muito curta para decomposição sazonal completa (precisa de pelo menos 24 meses para sazonalidade anual).")
    except Exception as e:
        st.error(f"Erro ao realizar decomposição sazonal: {e}")
        st.warning("Certifique-se de que os dados de tempo e vendas estão corretos e que a série tem dados suficientes.")

    return sales_ts, monthly_sales_avg, trend, seasonal, residual

def plot_seasonality_results(sales_ts, monthly_sales_avg, trend, seasonal, residual, title_suffix=""):
    """Plota os gráficos de sazonalidade no Streamlit."""
    if sales_ts is None or sales_ts.empty:
        st.warning(f"Não há dados para plotar a análise de sazonalidade {title_suffix}.")
        return

    st.subheader(f"Série Temporal de Vendas {title_suffix}")
    fig1, ax1 = plt.subplots(figsize=(24, 5))
    ax1.plot(sales_ts)
    ax1.set_title(f'Série Temporal de Vendas Agregadas Mensalmente {title_suffix}')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Valor de Venda')
    ax1.grid(True)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader(f"Padrão Sazonal Médio Mensal {title_suffix}")
    if monthly_sales_avg is not None and not monthly_sales_avg.empty:
        fig2, ax2 = plt.subplots(figsize=(24, 5))
        sns.barplot(x=monthly_sales_avg.index, y=monthly_sales_avg.values, palette='viridis', ax=ax2)
        ax2.set_title(f'Vendas Médias por Mês (Padrão Sazonal) {title_suffix}')
        ax2.set_xlabel('Mês')
        ax2.set_ylabel('Valor de Venda Médio')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader(f"Destaques da Sazonalidade {title_suffix}")
        
        peak_month_name = monthly_sales_avg.idxmax()
        peak_value = monthly_sales_avg.max()

        low_month_name = monthly_sales_avg.idxmin()
        low_value = monthly_sales_avg.min()

        st.info(f"**Mês de Pico de Vendas:** **{peak_month_name}** com vendas médias de **R$ {peak_value:,.2f}**.")
        st.info(f"**Mês de Baixa de Vendas:** **{low_month_name}** com vendas médias de **R$ {low_value:,.2f}**.")
    else:
        st.warning(f"Não foi possível calcular destaques de sazonalidade para o dataset {title_suffix}. Verifique os dados.")

    st.subheader(f"Componentes da Decomposição Sazonal {title_suffix}")
    if trend is not None and seasonal is not None and residual is not None:
        fig_decomp, axes = plt.subplots(4, 1, figsize=(24, 5), sharex=True)

        axes[0].plot(sales_ts)
        axes[0].set_title('Original')
        axes[0].grid(True)

        axes[1].plot(trend)
        axes[1].set_title('Tendência')
        axes[1].grid(True)

        axes[2].plot(seasonal)
        axes[2].set_title('Sazonalidade')
        axes[2].grid(True)

        axes[3].plot(residual)
        axes[3].set_title('Resíduo')
        axes[3].grid(True)

        plt.tight_layout()
        st.pyplot(fig_decomp)
        plt.close(fig_decomp)

        st.info("A **Tendência** mostra a direção geral das vendas.")
        st.info("A **Sazonalidade** revela o padrão repetitivo.")
        st.info("O **Resíduo** indica flutuações aleatórias ou ruído.")
    else:
        st.warning(f"A decomposição sazonal não pôde ser gerada para o dataset {title_suffix}. Isso pode ocorrer se a série temporal for muito curta ou não tiver dados suficientes para identificar um ciclo sazonal completo.")

# --- Carregar Resultados da Análise do Dataset Original ---
@st.cache_resource
def load_original_analysis_results():
    try:
        sales_ts = joblib.load(SALES_TS_SAVE_PATH)
        monthly_sales_avg = joblib.load(MONTHLY_SALES_SAVE_PATH)
        columns_mapping = joblib.load(COLUMNS_MAPPING_SAVE_PATH)

        trend = None
        seasonal = None
        residual = None
        if os.path.exists(DECOMPOSITION_SAVE_PATH_TREND) and \
           os.path.exists(DECOMPOSITION_SAVE_PATH_SEASONAL) and \
           os.path.exists(DECOMPOSITION_SAVE_PATH_RESIDUAL):
            trend = joblib.load(DECOMPOSITION_SAVE_PATH_TREND)
            seasonal = joblib.load(DECOMPOSITION_SAVE_PATH_SEASONAL)
            residual = joblib.load(DECOMPOSITION_SAVE_PATH_RESIDUAL)

        return sales_ts, monthly_sales_avg, columns_mapping, trend, seasonal, residual
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo de análise do dataset original não encontrado. Rode `analise_e_modelagem.py` primeiro para gerá-los.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os resultados da análise do dataset original: {e}")
        st.stop()

# --- Carregar o MELHOR Modelo e suas Features/Métricas ---
@st.cache_resource
def load_best_model():
    try:
        model = joblib.load(BEST_MODEL_SAVE_PATH)
        model_type = joblib.load(MODEL_TYPE_SAVE_PATH)
        feature_names = joblib.load(DT_FEATURES_SAVE_PATH) # Nomes das features são os mesmos para DT/RF/XGB
        metrics = joblib.load(MODEL_METRICS_SAVE_PATH) 
        return model, model_type, feature_names, metrics
    except FileNotFoundError:
        st.error(f"Erro: Arquivos do modelo de previsão não encontrados. Certifique-se de que '{BEST_MODEL_SAVE_PATH}', '{MODEL_TYPE_SAVE_PATH}', '{DT_FEATURES_SAVE_PATH}' e '{MODEL_METRICS_SAVE_PATH}' foram gerados pelo script de análise.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo de previsão: {e}")
        st.stop()

original_sales_ts, original_monthly_sales_avg, columns_mapping_loaded, \
original_trend, original_seasonal, original_residual = load_original_analysis_results()

best_regressor_model, best_model_type, model_feature_names, model_metrics = load_best_model()


# --- Layout da Aplicação com Abas ---
tab_sazonalidade_original, tab_sazonalidade_upload, tab_model_info, tab_previsao_vendas = st.tabs([
    "Sazonalidade (Original)",
    "Sazonalidade (Novo Dataset)",
    "Informações do Modelo",
    "Previsão de Vendas"
])

with tab_sazonalidade_original:
    st.header("Análise de Sazonalidade do Dataset Original")
    st.markdown("Esta seção mostra a análise de sazonalidade do dataset `dataset.csv`.")
    plot_seasonality_results(original_sales_ts, original_monthly_sales_avg, 
                             original_trend, original_seasonal, original_residual, "(Dataset Original)")

with tab_sazonalidade_upload:
    st.header("Analisar Novo Dataset para Sazonalidade")
    st.markdown("Faça upload de um arquivo CSV para analisar a sazonalidade em seus próprios dados de vendas.")
    st.info("**Importante:** O arquivo CSV deve conter colunas de 'Data do Pedido' e 'Valor de Venda'.")

    uploaded_file_sazonalidade = st.file_uploader("Escolha um arquivo CSV para análise de sazonalidade", type="csv", key="sazonalidade_uploader")

    if uploaded_file_sazonalidade is not None:
        try:
            # Delimitador AGORA é vírgula (,) para o NOVO dataset
            uploaded_df_raw_saz = pd.read_csv(uploaded_file_sazonalidade, sep=',', header=0)
            st.write("Pré-visualização do Dataset Carregado:")
            st.dataframe(uploaded_df_raw_saz.head())
            
            st.write(f"Dataset carregado com {len(uploaded_df_raw_saz)} linhas e {len(uploaded_df_raw_saz.columns)} colunas.")
            
            if st.button("Gerar Análise de Sazonalidade (Novo Dataset)", key="run_saz_analysis"):
                uploaded_sales_ts, uploaded_monthly_sales_avg, \
                uploaded_trend, uploaded_seasonal, uploaded_residual = \
                    perform_seasonality_analysis(uploaded_df_raw_saz.copy(), columns_mapping_loaded, 'Data_Pedido', 'Valor_Venda')
                
                if uploaded_sales_ts is not None and not uploaded_sales_ts.empty:
                    st.success("Análise de sazonalidade gerada com sucesso para o dataset carregado!")
                    plot_seasonality_results(uploaded_sales_ts, uploaded_monthly_sales_avg,
                                             uploaded_trend, uploaded_seasonal, uploaded_residual, "(Dataset Carregado)")
                else:
                    st.warning("Não foi possível realizar a análise de sazonalidade completa no dataset carregado. Verifique o formato das colunas 'Data do Pedido' e 'Valor de Venda' e se há dados suficientes.")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo CSV ou realizar a análise de sazonalidade: {e}")
            st.warning("Certifique-se de que o arquivo é um CSV válido, com o delimitador correto (vírgula) e com as colunas esperadas.")


with tab_model_info: # Aba renomeada
    st.header("Informações e Avaliação do Modelo de Previsão")
    st.markdown(f"O modelo de previsão de vendas mais eficaz encontrado é um **{best_model_type}**.")

    # Exibir as métricas de avaliação do modelo
    st.subheader("Métricas de Avaliação do Modelo")
    st.markdown(f"**RMSE (Root Mean Squared Error):** `R$ {model_metrics['rmse']:.2f}`")
    st.markdown(f"**MSE (Mean Squared Error):** `{model_metrics['mse']:.2f}`")
    st.markdown(f"**R² (Coeficiente de Determinação):** `{model_metrics['r2']:.2f}`")
    st.info("""
    * **RMSE:** Quanto menor, melhor. Representa o erro médio das previsões na mesma unidade do Valor de Venda.
    * **MSE:** Quanto menor, melhor. É o quadrado do RMSE, penaliza erros maiores.
    * **R²:** Quanto mais próximo de 1, melhor. Indica a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Um R² de 0.80 significa que 80% da variabilidade das vendas é explicada pelo modelo.
    """)

    # Exibir a imagem da árvore SOMENTE se o melhor modelo for uma DecisionTreeRegressor
    if best_model_type == "DecisionTreeRegressor":
        st.subheader("Visualização da Árvore de Decisão")
        st.markdown("Esta é a árvore de decisão que foi selecionada como o melhor modelo. Ela foi treinada para prever o **Valor de Venda** e pode servir como auxílio visual na compreensão das características do dataset.")
        if os.path.exists(TREE_IMAGE_PATH):
            st.image(TREE_IMAGE_PATH, caption="Árvore de Decisão do Modelo (Profundidade Limitada para Visualização)", use_container_width=True)
            st.info("A árvore exibe as regras que um modelo de regressão usa para estimar um valor. Cada nó representa uma condição em uma característica, e as folhas representam o valor médio previsto para as amostras que chegam àquele nó.")
        else:
            st.warning(f"A imagem da árvore de decisão ('{TREE_IMAGE_PATH}') não foi encontrada. Ela só é gerada se a Árvore de Decisão for o melhor modelo.")
    else:
        st.info(f"O melhor modelo selecionado é um **{best_model_type}**. Modelos de ensemble como o Random Forest ou XGBoost são compostos por muitas árvores e não possuem uma única árvore para visualização direta, mas geralmente oferecem maior robustez e eficácia.")


with tab_previsao_vendas: # Aba renomeada
    st.header("Faça uma Previsão de Venda com o Modelo de Vendas")
    st.markdown(f"Experimente diferentes valores para as características e veja qual o **Valor de Venda** previsto pelo modelo **{best_model_type}**.")

    # Obter os valores únicos para as colunas categóricas do dataset original
    try:
        # Carregar o DataFrame original para obter valores únicos
        df_original_full = pd.read_csv('dataset.csv', sep=',', header=0) # <-- MUDAR AQUI PARA VÍRGULA
        # Renomear as colunas (apenas as que existem no novo dataset)
        df_original_full.rename(columns={k: v for k, v in columns_mapping_loaded.items() if k in df_original_full.columns}, inplace=True)
        
        # Lista de colunas categóricas que você está usando como features
        # Baseado nas 'features_columns' definidas no analise_e_modelagem.py
        categorical_features_for_model_display = ['Segmento', 'Pais', 'Cidade', 'Estado', 'Categoria', 'SubCategoria']

        options = {}
        for col in categorical_features_for_model_display:
            if col in df_original_full.columns:
                # Tratar nulos e garantir que sejam strings antes de unique() e sorted()
                options[col] = [str(x) for x in df_original_full[col].dropna().unique()]
                # IMPORTANTE: Filtrar opções com muitas categorias (alta cardinalidade)
                # Para evitar dropdowns gigantes e problemas de performance/interface
                if len(options[col]) > 100: # Limite arbitrário, ajuste se necessário
                    st.warning(f"Coluna '{col}' tem muitas categorias ({len(options[col])}). Exibindo apenas as 100 mais frequentes para evitar sobrecarga na interface. Considere agrupá-las.")
                    top_categories = df_original_full[col].value_counts().nlargest(100).index.tolist()
                    options[col] = [str(x) for x in top_categories]
                if not options[col]: # Se a lista ainda estiver vazia após tratamento de nulos/filtragem
                    options[col] = [f"Nenhum Valor Encontrado para {col}"] # Fallback
            else:
                st.warning(f"Coluna '{col}' não encontrada no dataset original para preencher opções. Usando valores padrão.")
                # Fallbacks para colunas ausentes
                if col == 'Segmento': options[col] = ["Consumer", "Corporate", "Home Office"]
                elif col == 'Pais': options[col] = ["United States", "Canada", "Brazil"]
                elif col == 'Cidade': options[col] = ["New York", "Los Angeles", "Toronto"]
                elif col == 'Estado': options[col] = ["California", "New York", "Ontario"]
                elif col == 'Categoria': options[col] = ["Office Supplies", "Furniture", "Technology"]
                elif col == 'SubCategoria': options[col] = ["Art", "Chairs", "Phones"]
                else: options[col] = ["Default"]

    except Exception as e:
        st.error(f"Erro ao carregar o dataset original para obter opções de seleção: {e}")
        st.warning("Usando valores padrão para todas as opções de seleção.")
        options = {
            'Segmento': ["Consumer", "Corporate", "Home Office"],
            'Pais': ["United States", "Canada", "Brazil"],
            'Cidade': ["New York", "Los Angeles", "Toronto"],
            'Estado': ["California", "New York", "Ontario"],
            'Categoria': ["Office Supplies", "Furniture", "Technology"],
            'SubCategoria': ["Art", "Chairs", "Phones"],
        }

    # Controles de entrada para as features
    # AS COLUNAS NUMÉRICAS 'Quantidade', 'Desconto', 'Custo_Envio' NÃO ESTÃO MAIS DISPONÍVEIS
    col1, col2 = st.columns(2) # Reduzido para 2 colunas, pois há menos inputs

    with col1:
        segmento_input = st.selectbox("Segmento", sorted(options['Segmento']))
        pais_input = st.selectbox("País", sorted(options['Pais']))
        categoria_input = st.selectbox("Categoria", sorted(options['Categoria']))
    
    with col2:
        cidade_input = st.selectbox("Cidade", sorted(options['Cidade']))
        estado_input = st.selectbox("Estado", sorted(options['Estado']))
        subcategoria_input = st.selectbox("SubCategoria", sorted(options['SubCategoria']))

    st.markdown("---") # Separador visual

    # IMPORTANTE: Não há mais campos numéricos como Quantidade, Desconto, Custo_Envio
    # Se o modelo depender apenas de categóricas, a previsão será menos granular.
    # Se você quiser adicionar inputs numéricos, precisará de novas features numéricas no dataset.

    if st.button("Obter Previsão de Venda"):
        # Criar um DataFrame para a entrada do usuário
        input_data = pd.DataFrame({
            'Segmento': [segmento_input],
            'Pais': [pais_input],
            'Cidade': [cidade_input],
            'Estado': [estado_input],
            'Categoria': [categoria_input],
            'SubCategoria': [subcategoria_input],
            # As seguintes colunas NÃO estão no novo dataset e foram removidas do input_data:
            # 'Quantidade', 'Desconto', 'Custo_Envio', 'Prioridade_Pedido'
        })

        # Categoriais para One-Hot Encoding no Streamlit
        # Alinhado com as features usadas no analise_e_modelagem.py
        categorical_features_for_model = ['Segmento', 'Pais', 'Cidade', 'Estado', 'Categoria', 'SubCategoria']

        input_encoded = pd.get_dummies(input_data, columns=categorical_features_for_model)

        # Reindexar para garantir que todas as colunas de model_feature_names estejam presentes
        # e preencher com 0 onde não houver correspondência. Isso é CRÍTICO.
        input_processed = input_encoded.reindex(columns=model_feature_names, fill_value=0)

        try:
            predicted_sales = best_regressor_model.predict(input_processed)[0]
            st.success(f"O **Valor de Venda Previsto** pelo modelo é de: **R$ {predicted_sales:,.2f}**")
            st.info("Lembre-se que esta é uma previsão baseada nos padrões aprendidos pelo modelo e pode não refletir a realidade com 100% de precisão.")
            st.warning("Com menos features numéricas no dataset, a granularidade e a eficácia da previsão podem ser limitadas.")
        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")
            st.warning("Verifique se o modelo foi carregado corretamente e se as colunas de entrada correspondem às esperadas pelo modelo.")


st.markdown("---")
st.markdown("Desenvolvido por Jário Lima com Streamlit.")