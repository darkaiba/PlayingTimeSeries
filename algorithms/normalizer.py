from sklearn.preprocessing import MinMaxScaler
import polars as pl

class Normalizer:

    def __init__(self, json, dataframe):
        self.json = json
        self.dataframe = dataframe
        self.scaler = MinMaxScaler()

    def fit_transformer(self, dataframe, column):
        features = self.dataframe[column].to_numpy().reshape(-1, 1)
        return self.scaler.fit_transform(features)

    def inverse_transform(self, normalized_data):
        columns = self.dataframe.columns
        # Desnormaliza os dados
        datas_deanormalized = self.scaler.inverse_transform(normalized_data)
        return pl.DataFrame(datas_deanormalized, schema=columns)

    def split_datas_train(self, dataframe):
        columns = dataframe.columns
        #dataframe = dataframe.sample(fraction=1.0, shuffle=True, seed=42)

        train_size = int(len(dataframe) * self.json['test_size'])
        train = dataframe.slice(0, train_size)
        test = dataframe.slice(train_size, len(dataframe))

        if len(test) <= 0:
            test = None

        train = pl.DataFrame(train, schema=columns)
        if test is not None:
            test = pl.DataFrame(test, schema=columns)

        return train, test