import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
random.seed(42)
Faker.seed(42)

# Número de registros
total_registros = 50000

# Estrutura de localidade
localidades = {
    "Brazil": {
        "São Paulo": "SP",
        "Rio de Janeiro": "RJ",
        "Belo Horizonte": "MG",
        "Salvador": "BA",
        "Curitiba": "PR"
    },
    "United States": {
        "New York": "NY",
        "Los Angeles": "CA",
        "Chicago": "IL",
        "Houston": "TX",
        "Miami": "FL"
    },
    "Canada": {
        "Toronto": "ON",
        "Vancouver": "BC",
        "Montreal": "QC",
        "Calgary": "AB",
        "Ottawa": "ON"
    }
}

segmentos = ["Consumer", "Corporate", "Home Office"]
categorias_subcats = {
    "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
    "Office Supplies": ["Binders", "Paper", "Labels", "Storage"],
    "Technology": ["Phones", "Accessories", "Copiers", "Machines"]
}

# Valor médio e desvio padrão por categoria
valores_categoria = {
    "Furniture":     {"media": 1000, "desvio": 100},
    "Office Supplies": {"media": 100, "desvio": 50},
    "Technology":    {"media": 3000, "desvio": 200}
}

# Função para gerar ID parecido com o formato do dataset original
def gerar_id(prefixo):
    return f"{prefixo}-{random.randint(10000, 99999)}"

def gerar_dataset(n):
    dados = []
    for i in range(n):
        pais = random.choice(list(localidades.keys()))
        cidade = random.choice(list(localidades[pais].keys()))
        estado = localidades[pais][cidade]

        categoria = random.choice(list(categorias_subcats.keys()))
        subcat = random.choice(categorias_subcats[categoria])
        valor_raw = np.random.normal(
            loc=valores_categoria[categoria]["media"],
            scale=valores_categoria[categoria]["desvio"]
        )
        valor = round(max(1.0, valor_raw), 2)

        data_pedido = fake.date_between(start_date='-5y', end_date='today').strftime("%d/%m/%Y")
        id_pedido = f"{pais[:2].upper()}-{fake.year()}-{random.randint(100000,999999)}"
        id_cliente = gerar_id("CL")
        id_produto = gerar_id("PROD")

        dados.append([
            id_pedido,
            data_pedido,
            id_cliente,
            random.choice(segmentos),
            pais,
            cidade,
            estado,
            id_produto,
            categoria,
            subcat,
            valor
        ])

    colunas = [
        "ID_Pedido", "Data_Pedido", "ID_Cliente", "Segmento", "Pais",
        "Cidade", "Estado", "ID_Produto", "Categoria", "SubCategoria", "Valor_Venda"
    ]
    return pd.DataFrame(dados, columns=colunas)

# Gerar e salvar dataset
df_gerado = gerar_dataset(total_registros)
df_gerado.to_csv("dataset.csv", index=False)
