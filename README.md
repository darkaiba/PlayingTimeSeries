<h1>Séries Temporais</h1>
=================
<h2>Processo</h2>

<h4>Gerar a Base de Dados Fictícia de Séries Temporais</h4>
<p align="justify">Vamos criar uma base de dados sintética com uma coluna de Data e uma coluna de Valor. Os valores serão gerados a partir de uma função que combina tendências sazonais e aleatórias.</p>

<p align="justify">Exemplo de um arquivo de entrada (CSV):</p>
<!--ts-->

    Data,Valor
    2022-01-01,4.967141530112327
    2022-01-02,-1.2691572324556513
    2022-01-03,6.7038505625194
    2022-01-04,15.570730394321638
    2022-01-05,-1.8876543973730153
    2022-01-06,-1.774068203287789
    2022-01-07,16.472819662070457
    2022-01-08,8.468390693852742
    2022-01-09,-3.7873931742449134
    2022-01-10,6.446207427423555
    2022-01-11,-3.5003709664218388
    2022-01-12,-3.4103562959321776
    2022-01-13,3.7796291903950303
    2022-01-14,-17.659807125824994
    2022-01-15,-15.663276887488252
    2022-01-16,-3.9241568010617485
    2022-01-17,-8.316871048929022
    2022-01-18,5.066533432408331
    2022-01-19,-7.043668720595646
    2022-01-20,-11.974067379304472
    2022-01-21,16.917734297588055
    2022-01-22,0.1156336652811536
    2022-01-23,3.1606955882101837
    2022-01-24,-11.65019090838043
    2022-01-25,-2.734804595675228
    2022-01-26,3.9298282788533374
    2022-01-27,-8.577911859719652
    2022-01-28,6.800261207215068
    2022-01-29,-2.85201940155101

<!--te-->

<h4>Exploração e Análise Inicial dos Dados</h4>
<p align="justify">Vamos visualizar os dados gerados e realizar uma análise básica para entender o comportamento da série temporal.</p>
<p align="justify">Exibindo o gráfico da série temporal: O gráfico gerado mostrará como o valor da série evolui ao longo do tempo, com uma tendência geral de aumento, variações sazonais e ruído aleatório.</p>

<h4>Preparação dos Dados e do Modelo Preditivo</h4>
<p align="justify">Para a análise preditiva. Os dados serão preparados e normalizados para os processo de predição das séries temporais. O modelo será ajustado com base nos parametros de entrada no json.</p>
<p align="justify"></p>

<h4>Avaliação do Modelo</h4>
<p align="justify">O modelo foi ajustado aos dados de treinamento e usou as observações passadas para prever os valores futuros no conjunto de teste. O erro quadrático médio (MSE) é calculado para avaliar a precisão da previsão.</p>
<p align="justify">Se o MSE for baixo, significa que as previsões do modelo estão próximas dos valores reais.</p>

<h4>Geração de Relatório com os Gráficos e Métricas</h4>
<p align="justify">Será gerado um relatório. O gráfico resultante mostrará a série temporal original, os valores reais de teste e as previsões do modelo, permitindo uma visualização clara do desempenho do modelo. Também terá as métricas para uma melhor análise do modelo.</p>

<h2>Tipos de Algoritmos</h2>
=================

<h4>ARIAMA</h4>
<p align="justify">ARIMA é um modelo estatístico usado para análise e previsão de séries temporais. Ele captura padrões em dados que mudam ao longo do tempo.</p>
<ul>
    <li>Autoregressivo (AR): Usa valores passados da própria série para prever valores futuros.</li>
    <li>Integrado (I): Diferencia a série temporal (subtrai valores consecutivos) para torná-la estacionária (sem tendência ou sazonalidade).</li>
    <li>Média Móvel (MA): Usa erros de previsão passados para prever valores futuros.</li>
    <li>Uso: Adequado para séries temporais estacionárias ou que podem ser tornadas estacionárias.</li>
</ul>

<h4>Redes Neurais Recorrentes (RNN)</h4>
<p align="justify">RNNs são um tipo de rede neural projetada para processar sequências de dados, como séries temporais, texto ou áudio.</p>
<ul>
    <li>Possuem "memória" que permite que informações passadas influenciem o processamento de informações futuras.</li>
    <li>Usam loops para passar informações de um passo de tempo para o próximo.</li>
    <li>São muito boas em capturar dependências temporais.</li>
    <li>Uso: Previsão de séries temporais complexas, processamento de linguagem natural, reconhecimento de fala.</li>
</ul>

<h4>SARIAMAX</h4>
<p align="justify">SARIMAX é uma extensão do ARIMA que lida com sazonalidade e variáveis externas (regressores).</p>
<ul>
    <li>Incorpora componentes sazonais (SAR, SMA) para modelar padrões repetitivos.</li>
    <li>Permite incluir variáveis externas que podem influenciar a série temporal.</li>
    <li>É uma ferramenta muito poderosa para séries temporais que possuem sazonalidade e fatores externos que as influenciam.</li>
    <li>Uso: Previsão de vendas sazonais, demanda de energia, dados climáticos.</li>
</ul>

<h4>Suavização Exponencial</h4>
<p align="justify">ExponentialSmoothing é um conjunto de métodos para suavizar séries temporais e fazer previsões.</p>
<ul>
    <li>Atribui pesos exponencialmente decrescentes a observações passadas, dando mais importância a dados recentes.</li>
    <li>Existem variações (Holt, Holt-Winters) que lidam com tendências e sazonalidade.</li>
    <li>Holt-Winters, por exemplo, utiliza 3 fatores de suavização, um para o nível, um para a tendencia, e um para a sazonalidade.</li>
    <li>Uso: Previsões de curto prazo, suavização de dados ruidosos, detecção de tendências.</li>
</ul>

<h2>Como executar?</h2>
=================
<p align="justify">Para isso para executar o arquivo 'main.py'.</p>
<p align="justify">De entrada recebe um json com os seguintes parametros: </p>
<!--ts-->

    "input_file": True,                             ---> True ou False se o programa aceita um arquivo de entrada
    "reading": {                                    ---> Obrigatório. Parametro para leitura dos dados
        "reading_mode": 'csv',                      ---> Tipo de dados que vai ser lido (csv, json, parquet ou database)
        "host": None,                               ---> IP ou nome do host, pode ser nulo
        "user": None,                               ---> Nome de usuário, pode ser nulo
        "password": None,                           ---> Senha do usuário
        "path": 'C:\\datasets',                     ---> Diretório onde vai ler o arquivo de entrada
        "filename": 'dados_fakes_2.csv',            ---> Nome do arquivo de entrada
        "database": None,                           ---> Qual é o schema do banco de dados?
        "type_database": None                       ---> Qual é o tipo do banco de dados? (Mysql, Redshift, Postgres...)
    },
    "test_size": 0.9,                               ---> Qual é a porcentagem que você quer separar os dados para treino e teste (de 0.1 à 1.0)
    "algorithm": 'exponential',                     ---> Qual algoritmo você quer usar? (arima, sarimax, rnn ou exponential)
    "configs": {                                    ---> Configurações do algoritmo para gerar o modelo
        CADA ALGORITMO TEM SUA CONFIGURAÇÃO
    },
    "view": {                                       ---> Para visualização dos dados
        "column_value": 'Valor',                    ---> Nome da coluna que tem o valor das séries
        "colum_autocorrelation_partial": 'Valor',   ---> Nome da coluna que tem o valor das séries que você quer fazer a autocorrelação total
        "colum_autocorrelation": 'Valor',           ---> Nome da coluna que tem o valor das séries que você quer fazer a autocorrelação parcial
        "lags_autocorrelation_partial": 50,         ---> O valor que você quer fazer a autocorrelação total
        "lags_autocorrelation": 50,                 ---> O valor que você quer fazer a autocorrelação parcial
        "freq": 'D'                                 ---> Frequencia (D, M, Y)
    },
    "output_process":None                           ---> Pode ser nulo. Especificar a pasta de saida, senão vai salvar na pasta de execução.

<!--te-->

<p align="justify">Especificando a configuração no json para cada modelo:</p>

<h4>ARIMA</h4>
<p align="justify">
<!--ts-->

    "configs":{
        "column_value": 'Valor', #Nome da coluna que tem os valores da série
        "p": 30, #Um termo autoregressivo.
        "d": 2, #Uma diferenciação para tornar a série estacionária.
        "q": 10 #Um termo de média móvel.
    },

<!--te-->
</p>

<h4>RNN</h4>
<p align="justify">
<!--ts-->

     "configs": {
        "column_value": 'Valor', #Nome da coluna que tem os valores da série
        "epoch": 10, #Numero de épocas
        "validation_split": 0.25 #Para separar os dados na criaçao do modelo
        "n_camadas": 2, #número de camadas
        "return_sequences_enter": True, #Retorna sequências na camada de entrada (booleano: True ou False)
        "return_sequences_intermediate": False, #Retorna sequências nas camadas intermediárias (booleano: True ou False)
        "activation_enter": "tanh", #Função de ativação da camada de entrada (string: 'tanh', 'relu', 'sigmoid', etc.)
        "n_neuronio_enter": 64, # Número de neurônios de entrada, geralmente o número de dados que são passados
        "activation_intermediate": "relu", #Função de ativação da camada intermediária (string: 'tanh', 'relu', 'sigmoid', etc.)
        "n_neuronio_intermediate": 32, #Número de neurônios na camada intermediária
        "activation_end": "linear", #Função de ativação da camada de saída (string: 'tanh', 'relu', 'sigmoid', etc.)
        "n_neuronio_end": 1, #Número de neurônios de saída, geralmente o número de classes  
        "input_shape": [10, 1], #Formato dos dados de entrada (tupla de inteiros: passos de tempo, features)
        "optimizer": "adam", #Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        "loss": "mse", #Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        "metrics": ["mae"] #Métricas para avaliação (lista de strings - exemplo: accuracy)
    },

<!--te-->
</p>

<h4>SARIMAX</h4>
<p align="justify">
<!--ts-->

    "configs": {
        "column_value": 'Valor', #Nome da coluna que tem os valores da série
        "p": 30, #Um termo autoregressivo. Para ordem.
        "d": 2, #Uma diferenciação para tornar a série estacionária. Para ordem.
        "q": 10 #Um termo de média móvel. Para ordem.
        "a": 0, #Para ordem sazonal. Representa o número de termos autoregressivos sazonais
        "b": 0, #Para ordem sazonal. Representa o número de diferenciações sazonais
        "c": 0, #Para ordem sazonal. Representa o número de termos de média móvel sazonal
        "d": 0 #Para ordem sazonal. Representa o período da sazonalidade
    }

<!--te-->
</p>

<h4>Suavização Exponencial</h4>
<p align="justify">
<!--ts-->

    "configs": {
        "column_value": 'Valor', #Nome da coluna que tem os valores da série
        "trend": 'add', #Controla como a tendência (a direção geral dos dados, se estão subindo ou descendo)
        "seasonal": 'add', #Controla como a sazonalidade (padrões repetitivos dentro dos dados) é modelada.
        "seasonal_periods": 7 #Especifica o número de períodos em um ciclo sazonal.
    },

<!--te-->
</p>

<p align="justify">Ao final ele vai criar um pasta chamada "job_NOMEDOALGORITMO", com o relatório e imagens do modelo de predição.</p>

<p align="justify">Para mais detalhes acesse:</p>
<p align="justify"><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html">ARIMA</a></p>
<p align="justify"><a href="https://www.tensorflow.org/?hl=pt-br">Redes Neurais Recorrentes (TensowFlow)</a></p>
<p align="justify"><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html">SARIMAX</a></p>
<p align="justify"><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html">Suavização Exponencial</a></p>