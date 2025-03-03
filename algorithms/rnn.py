from statsmodels.tsa.stattools import adfuller
import polars as pl
from algorithms.view import View
from algorithms.normalizer import Normalizer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.report import Report
from datetime import datetime

class RNN:

    def __init__(self, json, dataframe, path_directory):
        self.json = json['configs']
        self.json_normalizer = json

        self.dataframe = dataframe
        self.dataframe_validacao = None
        self.dataframe_target = None
        self.dataframe_target_validacao = None

        self.view = View(self.json_normalizer['view'], dataframe, f"{path_directory}/imagem")
        self.normalizer = None

        self.path_directory = path_directory
        self.metrics = []

    def run(self):
        self.view.view_datas_nom("Rede Neural Recorrente (RNN)")
        self.test_stationarity()

        self.view.autocorrelation("Rede Neural Recorrente (RNN)")
        self.view.autocorrelation_partial("Rede Neural Recorrente (RNN)")

        self.prepare_datas()

        met = self.view.metrics
        nim = self.view.name_imagens
        self.view = View(self.json_normalizer['view'], self.dataframe, f"{self.path_directory}/imagem")
        self.view.metrics = met
        self.view.name_imagens = nim

        model, history = self.modeling()
        predict = self.predict(model)
        self.evaluate(model)

        #desanormalizar o predict
        df_predict = pl.DataFrame(predict, schema=[self.json['column_value']])
        normalizador_predict = Normalizer(self.json, df_predict)
        predict = normalizador_predict.fit_transformer(df_predict, self.json['column_value'])

        self.view.view_rnn_history(history, "Rede Neural Recorrente (RNN)")
        self.view.view_predicts(predict, self.dataframe_validacao, "Rede Neural Recorrente (RNN)")

        if self.dataframe_validacao is not None:
            column_value = self.json['column_value']
            self.view.view_predicts_validacao(self.dataframe, self.dataframe_validacao, predict, self.normalizer, normalizador_predict, column_value, "Rede Neural Recorrente (RNN)")

        self.metrics = self.metrics + self.view.metrics

        name = f"RNN_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
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
        epochs = self.json['epoch']
        validation_split = self.json['validation_split']
        # n_camadas - numero de camadas
        # return_sequences_enter - Retornar sequências na camada de entrada (booleano: True ou False)
        # return_sequences_intermediate - Retornar sequências nas camadas intermediárias (booleano: True ou False)
        # activation_enter - Função de ativação da camada de entrada (string: 'tanh', 'relu', 'sigmoid', etc.)
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros: passos de tempo, características)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        if self.json['n_camadas'] > 0:
            model.add(keras.layers.LSTM(self.json['n_neuronio_enter'],
                                        return_sequences=self.json['return_sequences_enter'],
                                        input_shape=self.json['input_shape']))

            for _ in range(
                    self.json['n_camadas'] - 2):  # -2 pois a primeira e a ultima camada são tratadas separadamente.
                model.add(keras.layers.LSTM(self.json['n_neuronio_intermediate'],
                                            return_sequences=self.json['return_sequences_intermediate']))

            if self.json['n_camadas'] > 1:
                model.add(keras.layers.LSTM(self.json['n_neuronio_intermediate']))
            else:
                model.add(keras.layers.LSTM(self.json['n_neuronio_intermediate'],
                                            return_sequences=(not self.json['return_sequences_intermediate'])))

        model.add(keras.layers.Dense(self.json['n_neuronio_end'], activation=self.json['activation_end']))
        model.compile(optimizer=self.json['optimizer'], loss=self.json['loss'], metrics=self.json['metrics'])

        if len(self.dataframe_target) <= 0:
            print(f"Dataframe Vazio. Precisa dos targets, o valor recebido foi de treino: {len(self.dataframe_target)}, validação: {len(self.dataframe_target_validacao)}")
            exit()

        return model, model.fit(self.dataframe, self.dataframe_target, epochs=epochs, validation_data=(self.dataframe_validacao, self.dataframe_target_validacao), verbose=0)

    def evaluate(self, model):
        print("Realizando as avaliações no modelo")
        perda_teste = model.evaluate(self.dataframe_validacao, self.dataframe_target_validacao, verbose=0)
        self.metrics.append(f"Perda no teste (Avaliação): {perda_teste}")

    def predict(self, model):
        print("Realizando as predições do modelo")
        return model.predict(self.dataframe_validacao).flatten()

    def prepare_datas(self):

        columns = self.dataframe.columns
        self.dataframe_target = pl.DataFrame(schema=columns)

        # Lidar com valores ausentes (Nulo e Nan)
        self.dataframe = self.dataframe.with_columns(pl.col(self.json['column_value']).fill_null(strategy="forward"))
        self.dataframe = self.dataframe.with_columns(pl.col(self.json['column_value']).fill_nan(0))

        # Normaliza os dados
        self.normalizer = Normalizer(self.json_normalizer, self.dataframe)
        dados_normalizados = self.normalizer.fit_transformer(self.dataframe, self.json['column_value'])

        datas, target = [], []
        for i in range(2, len(dados_normalizados)):

            datas.append(dados_normalizados[i-2:i-1])
            target.append(dados_normalizados[i-1:i])

        datas = np.concatenate(datas)
        target = np.concatenate(target)

        # Substituir as colunas originais pelos dados normalizados
        add_datas = {
            columns[0]: self.dataframe[columns[0]][len(datas)],
            columns[1]: pl.Series(self.json['column_value'], datas.flatten())
        }

        add_target = {
            columns[0] : self.dataframe[columns[0]][len(target)],
            columns[1] : pl.Series(self.json['column_value'], target.flatten())
        }

        self.dataframe = pl.DataFrame(add_datas)
        self.dataframe_target = pl.DataFrame(add_target)

        #Separa dados em treino e teste
        self.dataframe, self.dataframe_validacao = self.normalizer.split_datas_train(self.dataframe)
        self.dataframe_target, self.dataframe_target_validacao = self.normalizer.split_datas_train(self.dataframe_target)

        print(f"Dados de treino: {len(self.dataframe)}. Dados de validação: {len(self.dataframe_validacao)}")
        print(f"Dados de treino (target): {len(self.dataframe_target)}. Dados de validação (target): {len(self.dataframe_target_validacao)}")