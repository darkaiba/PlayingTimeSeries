import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from algorithms.normalizer import Normalizer
import numpy as np

class View:

    def __init__(self, json, dataframe, path_imagem):
        self.json = json
        self.dataframe = dataframe
        self.path_imagem = path_imagem
        self.name_imagens = []
        self.metrics = []

    def view_datas(self, name):
        print("Visualizando os dados")

        columns = self.dataframe.columns
        df_pandas = self.dataframe.to_pandas()

        name_image = f"Série_Temporal_Original_(Dados_Normalizados)-{name}.png"

        #df_pandas.plot(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        plt.plot(df_pandas[columns[0]], df_pandas[columns[1]], label='Dados históricos')
        plt.title(f'Série Temporal Original (Dados Normalizados) - {name}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.grid(True)
        plt.savefig(f"{self.path_imagem}/{name_image}")
        #plt.show()

        self.name_imagens.append(name_image)

    def view_datas_nom(self, name):
        print("Visualizando os dados")

        columns = self.dataframe.columns
        df_pandas = self.dataframe.to_pandas()

        name_image = f"Série_Temporal_Original_(Dados_Nao_Normalizados)-{name}.png"

        #df_pandas.plot(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        plt.plot(df_pandas[columns[0]], df_pandas[columns[1]], label='Dados históricos')
        plt.title(f'Série Temporal Original (Dados Não Normalizados) - {name}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.grid(True)
        plt.savefig(f"{self.path_imagem}/{name_image}")
        #plt.show()

        self.name_imagens.append(name_image)

    def autocorrelation(self, name):
        print("Visualizando os Autocorrelações")
        colum_autocorrelation = self.json['colum_autocorrelation']
        lags = self.json['lags_autocorrelation']

        name_image = f"Autocorrelação_ACF-{name}.png"

        plot_acf(self.dataframe[colum_autocorrelation], lags=lags)
        plt.title(f'Autocorrelação ACF - {name}')
        plt.savefig(f"{self.path_imagem}/{name_image}")
        self.name_imagens.append(name_image)
        #plt.show()

    def autocorrelation_partial(self, name):
        print("Visualizando os Autocorrelações Parciais")
        colum_autocorrelation = self.json['colum_autocorrelation_partial']
        lags = self.json['lags_autocorrelation_partial']

        name_image = f"Autocorrelação_PACF-{name}.png"

        plot_pacf(self.dataframe[colum_autocorrelation], lags=lags)
        plt.title(f'Autocorrelação Parcial PACF - {name}')
        plt.savefig(f"{self.path_imagem}/{name_image}")
        self.name_imagens.append(name_image)
        #plt.show()

    def view_predicts(self, predict, dataframe, name):
        print("Visualizando as Predições Originais")

        columns = dataframe.columns
        column_value = self.json['column_value']

        periods = len(predict)+1
        freq = self.json['freq']

        df_pandas = dataframe.to_pandas()

        name_image = f"Previsão_de_Séries_Temporais_(Dados_Normalizados)-{name}.png"

        plt.figure(figsize=(10, 6))
        plt.plot(df_pandas[column_value], label='Dados históricos')
        plt.plot(pd.date_range(df_pandas.index[-1], periods=periods, freq=freq)[1:], predict, color='red', label='Previsões', linestyle='--')
        plt.legend()
        plt.title(f'Previsão de Séries Temporais (Dados Normalizados) - {name}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.grid(True)
        plt.savefig(f"{self.path_imagem}/{name_image}")
        self.name_imagens.append(name_image)
        #plt.show()

    def view_rnn_history(self, history, name):
        name_image = f"Perda_durante_o_Treinamento-{name}.png"

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Treinamento')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title(f"Perda durante o Treinamento - {name}")
        plt.xlabel("Época")
        plt.ylabel("Perda (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.path_imagem}/{name_image}")
        self.name_imagens.append(name_image)
        # plt.show()

    def view_predicts_validacao(self, train_data, test_data, predictions, normalizador, normalizador_predict, column_value, name):
        print("Visualizando as Predições com dados de Validação e Predição com dados normalizados")

        print("Calculo de métricas")
        if len(test_data) == len(predictions):
            mse = mean_squared_error(test_data[column_value], predictions)
            mae = mean_absolute_error(test_data[column_value], predictions)
            r2 = r2_score(test_data[column_value], predictions)
            self.metrics.append(f'Erro Quadrado Médio (MSE): {mse}')
            self.metrics.append(f"Erro Absoluto Médio (MAE): {mae}")
            self.metrics.append(f"Coeficiente de Determinação (R²): {r2}")
        else:
            self.metrics.append(f"Não foi possivel calcular as métricas pois, tem tamanho diferentes. Dados validação: {len(test_data)}. Dados Predição: {len(predictions)}")

        print("Exibindo o gráfico")
        columns = self.dataframe.columns

        # Criar índices para os dados de treino e teste
        train_index = range(len(train_data))
        test_index = range(len(train_data), len(train_data) + len(test_data))

        name_image = f"Previsão_de_Série_Temporal_com_dados_de_Validação_e_de_Predição_(Dados_Normalizados)-{name}.png"

        # Visualizando as previsões
        plt.figure(figsize=(10, 6))
        plt.plot(train_index, train_data[column_value], label='Treinamento', color='blue')
        plt.plot(test_index, test_data[column_value], label='Teste Real', color='green')

        if len(test_data) == len(predictions):
            plt.plot( test_index, predictions, label='Previsões', color='red', linestyle='--')

        plt.title(f'Previsão de Série Temporal com dados de Validação e de Predição (Dados Normalizados) - {name}')
        plt.xlabel("Indice")
        plt.ylabel(columns[1])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.path_imagem}/{name_image}")
        self.name_imagens.append(name_image)
        #plt.show()
