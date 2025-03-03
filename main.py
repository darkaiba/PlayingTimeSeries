from src.getdatas import DataReaderFile, DataReaderRemote
from src import Algoritmo
from algorithms.forecast import Forecast
from algorithms.rnn import RNN
from algorithms.sarimax import ForecastSarimax
from algorithms.exponential import ForecastExpo
import os

def main(json_enter):

    dataframe = None
    if json_enter['input_file'] is False:
        # Busca dados em um servidor remoto ou banco de dados
        print("Buscando os dados")
        reader = DataReaderRemote(json_enter)
    elif json_enter['input_file'] is True:
        # Faz a leitura do arquivo de entrada (Local)
        print("Lendo o Arquivo de Entrada")
        reader = DataReaderFile(json_enter)
        dataframe = reader.read_data()
    else:
        ValueError(f"Parametro não é válido, 'input_file' --> {json_enter['input_file']}")

    diretorio_script = None
    if json_enter['output_process'] is None:
        diretorio_script = os.path.dirname(os.path.abspath(__file__))
    else:
        diretorio_script = os.path.dirname(json_enter['output_process'])

    diretorio_script = create_dir(diretorio_script, json_enter)

    if json_enter['algorithm'] == Algoritmo.ARIMA:
        print("iniciando predição com ARIMA")
        Forecast(json_enter, dataframe, diretorio_script).run()
    elif json_enter['algorithm'] == Algoritmo.RNN:
        print("iniciando predição com Rede Neural Recorrente (RNN)")
        RNN(json_enter, dataframe, diretorio_script).run()
    elif json_enter['algorithm'] == Algoritmo.SARIMAX:
        print("iniciando predição com Sarimax")
        ForecastSarimax(json_enter, dataframe, diretorio_script).run()
    elif json_enter['algorithm'] == Algoritmo.EXPONENTIAL:
        print("iniciando predição com Exponential")
        ForecastExpo(json_enter, dataframe, diretorio_script).run()
    else:
        ValueError(f"Parametro não é válido, 'algorithm' --> {json_enter['algorithm']}")

def create_dir(diretorio_script, json_enter):

    print("Preparando os diretórios")
    job = os.path.join(diretorio_script, f'job_{json_enter['algorithm']}')
    relatorio = os.path.join(job, 'relatorio')
    imagens = os.path.join(job, 'imagem')

    if not os.path.exists(job):
        try:
            os.makedirs(job)
            print(f"Diretório '{job}' criado com sucesso.")
        except OSError as erro:
            print(f"Erro ao criar o diretório '{job}': {erro}")

    if not os.path.exists(relatorio):
        try:
            os.makedirs(relatorio)
            print(f"Diretório '{relatorio}' criado com sucesso.")
        except OSError as erro:
            print(f"Erro ao criar o diretório '{relatorio}': {erro}")

    if not os.path.exists(imagens):
        try:
            os.makedirs(imagens)
            print(f"Diretório '{imagens}' criado com sucesso.")
        except OSError as erro:
            print(f"Erro ao criar o diretório '{imagens}': {erro}")

    return job

if __name__ == "__main__":
    print("Iniciando o processo de previsão de séries temporais!")

    json_enter_arima = {
        "input_file":True,
        "reading": {
            "reading_mode": 'csv',
            "host": None,
            "user": None,
            "password": None,
            "path": 'C:\\Users\\ph_li\\PycharmProjects\\TimeSeries\\.venv\\datasets',
            "filename": 'dados_fakes_2.csv',
            "database": None,
            "type_database": None
        },
        "test_size": 0.9,
        "algorithm": 'arima',
        "configs":{
            "column_value": 'Valor',
            "p": 30, #Um termo autoregressivo.
            "d": 2, #Uma diferenciação para tornar a série estacionária.
            "q": 10 #Um termo de média móvel.
        },
        "view":{
            "column_value": 'Valor',
            "colum_autocorrelation_partial": 'Valor',
            "colum_autocorrelation": 'Valor',
            "lags_autocorrelation_partial": 50,
            "lags_autocorrelation": 50,
            "freq": 'D'
        },
        "output_process":None
    }

    json_enter_rnn = {
        "input_file": True,
        "reading": {
            "reading_mode": 'csv',
            "host": None,
            "user": None,
            "password": None,
            "path": 'C:\\Users\\ph_li\\PycharmProjects\\TimeSeries\\.venv\\datasets',
            "filename": 'dados_fakes_2.csv',
            "database": None,
            "type_database": None
        },
        "test_size": 0.75,
        "algorithm": 'rnn',
        "configs": {
            "column_value": 'Valor',
            "epoch": 10,
            "validation_split": 0.25,
            "n_camadas": 2,
            "return_sequences_enter": True,
            "return_sequences_intermediate": False,
            "activation_enter": "tanh",
            "n_neuronio_enter": 64,
            "activation_intermediate": "relu",
            "n_neuronio_intermediate": 32,
            "activation_end": "linear",
            "n_neuronio_end": 1,
            "input_shape": [10, 1],
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["mae"]
        },
        "view": {
            "column_value": 'Valor',
            "colum_autocorrelation_partial": 'Valor',
            "colum_autocorrelation": 'Valor',
            "lags_autocorrelation_partial": 50,
            "lags_autocorrelation": 50,
            "freq": 'D'
        },
        "output_process":None
    }

    json_enter_sarimax = {
        "input_file": True,
        "reading": {
            "reading_mode": 'csv',
            "host": None,
            "user": None,
            "password": None,
            "path": 'C:\\Users\\ph_li\\PycharmProjects\\TimeSeries\\.venv\\datasets',
            "filename": 'dados_fakes_2.csv',
            "database": None,
            "type_database": None
        },
        "test_size": 0.75,
        "algorithm": 'sarimax',
        "configs": {
            "column_value": 'Valor',
            "p": 30,  # Um termo autoregressivo.
            "d": 2,  # Uma diferenciação para tornar a série estacionária.
            "q": 10,  # Um termo de média móvel.
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0
        },
        "view": {
            "column_value": 'Valor',
            "colum_autocorrelation_partial": 'Valor',
            "colum_autocorrelation": 'Valor',
            "lags_autocorrelation_partial": 50,
            "lags_autocorrelation": 50,
            "freq": 'D'
        },
        "output_process":None
    }

    json_enter_expo = {
        "input_file": True,
        "reading": {
            "reading_mode": 'csv',
            "host": None,
            "user": None,
            "password": None,
            "path": 'C:\\Users\\ph_li\\PycharmProjects\\TimeSeries\\.venv\\datasets',
            "filename": 'dados_fakes_2.csv',
            "database": None,
            "type_database": None
        },
        "test_size": 0.9,
        "algorithm": 'exponential',
        "configs": {
            "column_value": 'Valor',
            "trend": 'add',
            "seasonal": 'add',
            "seasonal_periods": 7
        },
        "view": {
            "column_value": 'Valor',
            "colum_autocorrelation_partial": 'Valor',
            "colum_autocorrelation": 'Valor',
            "lags_autocorrelation_partial": 50,
            "lags_autocorrelation": 50,
            "freq": 'D'
        },
        "output_process":None
    }

    #main(json_enter_arima)
    main(json_enter_rnn)
    #main(json_enter_sarimax)
    #main(json_enter_expo)