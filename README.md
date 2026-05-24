# Relatório Técnico Relatório Técnico - Precificação de Imóveis (Ames Housing)

## 1. Capa
* **Equipe:** Grupo G
* **Integrantes:**
  * Gabriel Abreu Cunha De Alencar - 2315097
  * Igor Gomes Ximenes -2217665
  * Kalil Smith Pinto Palheta - 2223857

* **Repositório:**
    * [Repositório de Submissão](https://github.com/Kalil-S/Repositorio-de-Submissao-de-Precos-de-Imoveis-.git)
    * [Repositório de Desenvolvimento](https://github.com/Kalil-S/Repositorio-de-Desenvolvimento-de-Precos-de-Imoveis.git)

---

## 2. Introdução
Este projeto detalha os procedimentos metodológicos, estatísticos e computacionais adotados no desenvolvimento de um modelo preditivo para a base de dados *Ames Housing*. O cerne do problema consistiu em estimar o valor de venda (`SalePrice`) de propriedades residenciais por intermédio de múltiplos atributos estruturais, espaciais e temporais. A resolução de problemas de regressão no contexto imobiliário demanda um tratamento rigoroso da alta dimensionalidade de características contínuas e categóricas.

Para fins de otimização paramétrica, a métrica fundamental estabelecida pela equipe foi o **RMSLE (Root Mean Squared Logarithmic Error)**, mas não deixando de atender as outra metricas registradas no arquivo "metricas_baseline.txt". A escolha dessa métrica justifica-se estatisticamente pois a avaliação do erro em escala logarítmica garante que a penalização incida sobre a proporção do desvio em relação ao valor intrínseco da propriedade, mitigando severas distorções na função de custo causadas por *outliers* naturais (casas de luxo com preços exponencialmente maiores).

---

## 3. Análise Exploratória de Dados (EDA)
A etapa de exploração visual permitiu identificar a estrutura latente dos dados e balizar as decisões de engenharia. 
* **Análise de Densidade da Variável Alvo**A avaliação do histograma de densidade empírica da variável dependente `SalePrice` — documentada na figura abaixo(`eda_target_distribution.png`) — comprovou que a distribuição original dos preços sofre de uma acentuada assimetria positiva à direita (*Right Skewness*). Sob a ótica estatística, o desvio de uma distribuição gaussiana viola o princípio de homocedasticidade dos resíduos exigido por estimadores lineares clássicos. A identificação visual dessa cauda longa à direita justifica matematicamente a necessidade de aplicar uma transformação logarítmica sobre o alvo, estabilizando a variância e aproximando a distribuição do formato normal.
<img src="https://github.com/Kalil-S/Repositorio-de-Desenvolvimento-de-Precos-de-Imoveis/blob/main/plots/eda_target_distribution.png">

* **Identificação de Outliers:** Histogramas de densidade comprovaram que a distribuição da variável alvo (`SalePrice`) sofre de assimetria positiva substancial (*Right Skewness*), uma característica esperada em ativos financeiros. Gráficos de dispersão (*scatter plots*) evidenciaram a presença de propriedades com vasta área territorial, mas valores atípicos de venda, identificando *outliers* primários.
<img src="https://github.com/Kalil-S/Repositorio-de-Desenvolvimento-de-Precos-de-Imoveis/blob/main/plots/eda_engineered_features.png">

* **Identificação de Correlações Relevantes:** A matriz de correlação de Pearson destacou que atributos dimensionais globais e índices de qualidade construtiva ostentam as mais elevadas correlações lineares positivas e diretas com o preço. Variáveis fragmentadas ou esparsas isoladamente demonstraram baixo coeficiente de determinação explicativo.
<img src ="https://github.com/Kalil-S/Repositorio-de-Desenvolvimento-de-Precos-de-Imoveis/blob/main/plots/eda_correlation.png">

---

## 4. Pré-processamento e Feature Engineering
Para preparar a base para os algoritmos matemáticos e evitar *data leakage* (vazamento de dados), consolidamos todas as transformações na classe customizada `AmesFeatureEngineer`.

* **Tratamento de Valores Faltantes e Outliers:** Lidamos com a ausência de dados utilizando o `SimpleImputer` e preenchimentos lógicos com `fillna(0)` para propriedades que não possuíam determinadas áreas (ex: casas sem porão).
* **Transformações Aplicadas:** Utilizamos o `StandardScaler` para escalonar e normalizar as variáveis numéricas, garantindo que variáveis com grandezas muito diferentes não enviesassem o modelo. Para as variáveis categóricas, aplicamos o `OneHotEncoder` de forma integrada no `ColumnTransformer`.
* **Novas Variáveis Criadas:**
  * **`TotalBath`:** Consolidamos os banheiros criando uma métrica única que soma os banheiros completos e os lavabos (tanto do andar principal quanto do porão), atribuindo o peso de `0.5` para os lavabos (`HalfBath` e `BsmtHalfBath`).
  * **`TotalSF`:** Criamos a variável de Área Construída Total, somando a área do porão (`TotalBsmtSF`), do primeiro andar (`1stFlrSF`) e do segundo andar (`2ndFlrSF`), agrupando as informações espaciais em uma *feature* de alto poder preditivo.

---

## 5. Modelagem e Validação
* **Algoritmos Testados:** Foram testados algoritmos de diferentes complexidades, passando por modelos lineares até métodos baseados em árvores e *ensembles*, como **Regressão Linear Múltipla**, **Random Forest** e **XGBoost** (`XGBRegressor`).
* **Técnica de Validação:** Realizamos um *Holdout* inicial separando a base em 80% para treino e 20% para teste. Para garantir a robustez na etapa de treino e evitar o *overfitting* (sobreajuste), aplicamos a técnica de **Validação Cruzada** (*Cross-Validation*), avaliando o poder de generalização do modelo em diferentes subconjuntos dos dados.
* **Busca de Hiperparâmetros:** Os melhores hiperparâmetros foram definidos através de busca automatizada (como o *GridSearchCV*), ajustando parâmetros essenciais como a profundidade máxima das árvores e a taxa de aprendizado (*learning rate*) no XGBoost, focando na minimização do erro.

---
## 6. Conclusões

### Tabela Comparativa de Performance
Os resultados subsequentes refletem o desempenho computado durante a fase rigorosa de **Validação Cruzada pós-tuning**, onde as metodologias demonstraram seus erros e desvios reais:

| Modelo de Machine Learning | R² Médio | R² Desvio | RMSLE Médio | RMSLE Desvio | MAE Médio | MAE Desvio | Tempo de Treino (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge (L2)** | **0,93180** | 0,00644 | **0,10843** | 0,00811 | **$ 13.157,10** | $ 655,29 | **0,32s** |
| **XGBoost** | 0,90904 | 0,00871 | 0,11851 | 0,00736 | $14.294,88 |$ 1.347,00 | 2,89s |
| **Linear Regression** | 0,89387 | 0,01449 | 0,13429 | 0,01273 | $15.698,01 |$ 1.269,05 | 0,38s |
| **Random Forest** | 0,88381 | 0,00741 | 0,13444 | 0,00934 | $16.363,51 |$ 1.242,87 | 24,19s |

### Justificativa de Erro e Impacto no Negócio
Nosso modelo computacional final baseado na regressão Ridge obteve um Erro Médio Absoluto (MAE) estipulado em cerca de **$ 13.157,10**. Isso significa que a máquina erra a precificação das residências, em média, por 13,1 mil dólares, tanto subestimando quanto superestimando o laudo predial. 

Considerando um mercado imobiliário padrão cujos imóveis ultrapassam o crivo de $200.000, um desvio absoluto de$ 13 mil enquadra-se dentro de uma variância analítica excepcional (inferior a 7%). Este nível de estabilidade estatística confere extrema solidez para o negócio, permitindo mitigar o risco humano de avaliações viesadas e agilizando as operações em sistemas digitais voltados a imobiliárias de alto volume.

### Justificativa Final
A decisão fundamentada pela arquitetura matemática do **Ridge (Regularização L2)** comprova-se cientificamente pelo conjunto de métricas aferidas. O *tuning* paramétrico do Ridge (com `alpha: 25.0`) conseguiu resolver eficientemente os efeitos colaterais da maldição da dimensionalidade provocada pela vasta quantia de colunas do *OneHotEncoding*. 

O Ridge distanciou-se estatisticamente do então campeão presumido (XGBoost) ao elevar a taxa de explicação de variância (R² médio em 0,9318) e reduzir drasticamente a perda logarítmica em ambiente de Cross-Validation (RMSLE de 0,10843) com pouquíssima oscilação. Adicionalmente, seu ciclo de validação foi completo em levíssimos **0,32 segundos**, consolidando uma robusta relação custo-benefício computacional. Um modelo infinitamente mais ágil, computacionalmente enxuto, e substancialmente mais exato na métrica final da competição do que complexas redes de *ensembles*.
