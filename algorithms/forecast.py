from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import polars as pl
from algorithms.view import View
from algorithms.normalizer import Normalizer
from src.report import Report
from datetime import datetime

class Forecast:

    def __init__(self, json, dataframe, path_directory):
        self.json = json['configs']
        self.json_normalizer = json

        self.dataframe = dataframe
        self.dataframe_validacao = None

        self.view = None
        self.normalizer = None

        self.path_directory = path_directory
        self.metrics = []

    def run(self):
        self.prepare_datas()
        self.view = View(self.json_normalizer['view'], self.dataframe, f"{self.path_directory}/imagem")

        self.view.view_datas("ARIMA")

        self.test_stationarity()

        self.view.autocorrelation("ARIMA")
        self.view.autocorrelation_partial("ARIMA")

        model = self.modeling()
        predict = self.predict(model)

        df_predict = pl.DataFrame(predict, schema=[self.json['column_value']])
        normalizador_predict = Normalizer(self.json, df_predict)
        #predict = normalizador_predict.fit_transformer(df_predict, self.json['column_value'])

        self.view.view_predicts(predict, self.dataframe_validacao, "ARIMA")

        if self.dataframe_validacao is not None:
            column_value = self.json['column_value']
            self.view.view_predicts_validacao(self.dataframe, self.dataframe_validacao, predict, self.normalizer, None, column_value, "ARIMA")

        self.metrics = self.metrics + self.view.metrics
        self.view_model(model)

        name = f"Arima_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
        report = Report()
        pdf_report = report.create_report(name, self.view.name_imagens, self.path_directory, self.metrics)
        report.save_report(pdf_report)

    def test_stationarity(self):
        print("Verificando se é uma série estacionária")
        colum_stationary = self.json['column_value']

        result = adfuller(self.dataframe[colum_stationary])
        self.metrics.append(f'Estacionariedade: {result[1]}')

        # Se o valor de p for menor que 0.05, podemos concluir que a série é estacionária
        if result[1] < 0.05:
            self.metrics.append("É uma série estacionária!")

    def modeling(self):
        print("Criando o modelo e fazendo o treinamento")
        column_value = self.json['column_value']
        p = self.json['p']
        d = self.json['d']
        q = self.json['q']

        model = ARIMA(self.dataframe[column_value].to_numpy(), order=(p, d, q))
        return model.fit()

    def predict(self, model):
        print("Realizando as predições do modelo")
        n_steps = len(self.dataframe_validacao)
        return model.forecast(steps=n_steps)

    def view_model(self, model):
        # Exibir o resumo do modelo
        self.metrics.append(str(model.summary()))

        n_steps = len(self.dataframe_validacao)
        # Prever os próximos valores
        previsoes = model.get_forecast(steps=n_steps)
        previsoes_medias = previsoes.predicted_mean
        intervalo_confianca = previsoes.conf_int()

    def prepare_datas(self):

        # Lidar com valores ausentes (Nulo e Nan)
        self.dataframe = self.dataframe.with_columns(pl.col(self.json['column_value']).fill_null(strategy="forward"))
        self.dataframe = self.dataframe.with_columns(pl.col(self.json['column_value']).fill_nan(0))

        # Normaliza os dados
        self.normalizer = Normalizer(self.json_normalizer, self.dataframe)
        dados_normalizados = self.normalizer.fit_transformer(self.dataframe, self.json['column_value'])

        # Substituir as colunas originais pelos dados normalizados
        self.dataframe = self.dataframe.with_columns([
            pl.Series(self.json['column_value'], dados_normalizados.flatten()),
        ])

        #Separa dados em treino e teste
        self.dataframe, self.dataframe_validacao = self.normalizer.split_datas_train(self.dataframe)
        print(f"Dados de treino: {len(self.dataframe)}. Dados de validação: {len(self.dataframe_validacao)}")
