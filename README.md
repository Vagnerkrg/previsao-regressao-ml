# 📈 Previsão e Precificação de Imóveis com Regressão Linear

🐍 Python 📦 v1.0.0

Este projeto desenvolve um modelo estatístico preditivo baseado em machine learning para estimar o preço de imóveis com base em suas características estruturais e localização urbana.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python
* **Análise de Dados:** `pandas`, `numpy`
* **Modelagem Estatística:** `statsmodels` (Mínimos Quadrados Ordinários - OLS)
* **Visualização:** `matplotlib`, `seaborn`
* **Persistência de Objetos:** `pickle`

---

## 📊 Estrutura e Desenvolvimento do Modelo

### 1. Transformações e Escala dos Dados
Para corrigir distorções nas distribuições das variáveis contínuas, aplicamos transformações matemáticas com escala logarítmica:
* `np.log()`: Aplicada às áreas e distâncias.
* `np.log1p()`: Aplicada à área do quintal para gerenciar valores zerados sem quebras matemáticas.

### 2. Modelagem Estatística (OLS)
Utilizamos a biblioteca `statsmodels` para construir uma regressão linear robusta. Incluímos o cálculo do intercepto (Beta 0) adicionando uma constante aos dados de entrada (`sm.add_constant`). A validação do modelo foi feita através de métricas chaves extraídas do sumário estatístico:
* **R² (Coeficiente de Determinação):** Para avaliar a qualidade global do ajuste.
* **P-valores (Significância Individual):** Para garantir que apenas variáveis estatisticamente relevantes (p < 0.05) fossem mantidas.
* **Análise Estatística de Resíduos:** Verificação de normalidade e homocedasticidade.

### 3. Interpretação Econômica de Variáveis Dummy
Calculamos o impacto percentual exato da variável categórica `existe_segundo_andar` utilizando a reversão exponencial específica para coeficientes log-lineares:
$$\text{Efeito \%} = 100 \times (e^{\beta} - 1)$$
Implementado em código através da função `np.expm1()`.

### 4. Salvamento e Previsões em Lote
O modelo final treinado foi serializado em um arquivo binário `.pkl` utilizando a biblioteca `pickle`. Com isso, estruturamos um pipeline estável que carrega o modelo em produção, processa novos dados brutos (`casas_a_precificar.csv`), injeta as variáveis na mesma escala do treino e reverte a saída final usando `np.exp()` para exibir as estimativas finais diretamente em Reais (R$).

---

## 📂 Estrutura do Repositório

* `Dados/`: Contém os arquivos de dados originais e as bases para precificação.
* `Modelos/`: Armazena o arquivo serializado contendo o modelo treinado `.pkl`.
* `transformacao_vars.ipynb`: Notebook com a exploração de dados, engenharia de atributos e validações.
* `requirements.txt`: Arquivo com todas as dependências do ambiente virtual (`venv`).
