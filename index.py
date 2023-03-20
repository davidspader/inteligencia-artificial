import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv("barcos_ref.csv")

sns.heatmap(tabela.corr()[["Preco"]], annot=True, cmap="Blues")
plt.show()

y = tabela["Preco"]
x = tabela.drop("Preco", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

modeloRegressaoLinear = LinearRegression()
modeloArvoreDecisao = RandomForestRegressor()

modeloRegressaoLinear.fit(x_treino, y_treino)
modeloArvoreDecisao.fit(x_treino, y_treino)
RandomForestRegressor()

previsaoRegressaoLinear = modeloRegressaoLinear.predict(x_teste)
previsaoArvoreDecisao = modeloArvoreDecisao.predict(x_teste)

print(metrics.r2_score(y_teste, previsaoRegressaoLinear))
print(metrics.r2_score(y_teste, previsaoArvoreDecisao))

tabelaAuxiliar = pd.DataFrame()
tabelaAuxiliar["y_teste"] = y_teste
tabelaAuxiliar["Previsoes Arvore Decisão"] = previsaoArvoreDecisao
tabelaAuxiliar["Previsoes Regressão linear"] = previsaoRegressaoLinear

sns.lineplot(data=tabelaAuxiliar)
plt.show()

novaTabela = pd.read_csv("novos_barcos.csv")
previsao = modeloArvoreDecisao.predict(novaTabela)
print(previsao)