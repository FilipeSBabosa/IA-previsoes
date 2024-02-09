# passo a passo
# passo 0 - entender o desifio da empresa
# passo 1 - importar a base de dados
# passo 2 - preparar a base de dados para a IA
# PASSO 3 - criar um modelo de IA » Score de créditos: ruim, médio, bom
# passo 4 - escolher o melhor modelo
# passo 5 - usar a nossa IA para fazer novas precisões
#!pip install scikit-learn
import pandas as pd

tabela = pd.read_csv("clientes.csv")
print(tabela)
print(tabela.info())
print(tabela.columns)
# profisao
# mix_credito
# comportamento_pagamento
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])
print(tabela.info())
# aprendizado de máquina
# y é a coluna que mais quer prever
# x são as colunas que você vai usar para fazer previsão 
    # não vamos usar a coluna id_cliente porque ela é um número aleatório

x = tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3 , random_state=1)
# criar a inteligencia artificial
# arvore de decisão » RandomForest
# KW » vizinhos Próximos » kneightbors
# importa IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# criar IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
# treinar a IA 
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
# testar os modelos 3 comparar os modelos 
# acurância
from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsao_knn))
# melhor modelo: medelo_arvoredecisao
# fazer novas previsões
# importar novos clientes
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
print(tabela_novos_clientes)
# codificar os novos clientes
# codificador » aplica na coluna "profissao"
tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])
# codificador » aplica na coluna "mix_credito"
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])
# codificador » aplica na coluna "comportamento_pagamento"
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])
# fazer as previsões
previsoes = modelo_arvoredecisao.predict(tabela_novos_clientes)
print(previsoes)