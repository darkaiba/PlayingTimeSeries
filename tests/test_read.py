import unittest
from unittest.mock import patch, MagicMock
import polars as pl
from io import BytesIO
from src.getdatas import DataReaderRemote, DataReaderFile
from src import TypeFile

class TestDataReaderRemote(unittest.TestCase):

    def setUp(self):
        """Configuração inicial para os testes."""
        self.config_csv = {
            "reading": {
                "reading_mode": TypeFile.CSV,
                "host": "example.com",
                "user": "user",
                "password": "password",
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": "iris_dataset_csv.csv"
            }
        }

    @patch('src.getdatas.AcessServer')
    def test_read_csv(self, mock_acess_server):
        """Testa a leitura de um arquivo CSV."""
        # Configura o mock
        mock_instance = mock_acess_server.return_value
        mock_instance.get_file_chunks.return_value = [b"col1,col2\n1,2\n3,4\n"]

        # Cria a instância de DataReaderRemote
        reader = DataReaderRemote(self.config_csv)

        # Lê os dados
        result = list(reader.read_data(chunk_size=1024))

        # Verifica o resultado
        expected_df = pl.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].equals(expected_df))

    def test_invalid_data_type(self):
        """Testa o comportamento com um tipo de leitura inválido."""
        invalid_config = {
            "reading": {
                "reading_mode": "invalid_type",
                "host": "example.com",
                "user": "user",
                "password": "password",
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": "data.invalid"
            }
        }
        reader = DataReaderRemote(invalid_config)
        with self.assertRaises(ValueError):
            list(reader.read_data(chunk_size=1024))

class TestDataReaderFile(unittest.TestCase):

    def setUp(self):
        """Configuração inicial para os testes."""
        self.config_csv = {
            "reading": {
                "reading_mode": TypeFile.CSV,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": "iris_dataset_csv.csv"
            }
        }
        self.config_json = {
            "reading": {
                "reading_mode": TypeFile.JSON,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": "iris_dataset_json.json"
            }
        }
        self.config_parquet = {
            "reading": {
                "reading_mode": TypeFile.PARQUET,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": "iris_dataset_parquet.parquet"
            }
        }

    @patch('polars.read_csv')
    def test_read_csv(self, mock_read_csv):
        """Testa a leitura de um arquivo CSV."""
        # Configura o mock
        expected_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = expected_df

        # Cria a instância de DataReaderFile
        reader = DataReaderFile(self.config_csv)

        # Lê os dados
        result = reader.read_data()

        # Verifica o resultado
        self.assertTrue(result.equals(expected_df))

    @patch('polars.read_json')
    def test_read_json(self, mock_read_json):
        """Testa a leitura de um arquivo JSON."""
        # Configura o mock
        expected_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_json.return_value = expected_df

        # Cria a instância de DataReaderFile
        reader = DataReaderFile(self.config_json)

        # Lê os dados
        result = reader.read_data()

        # Verifica o resultado
        self.assertTrue(result.equals(expected_df))

    @patch('polars.read_parquet')
    def test_read_parquet(self, mock_read_parquet):
        """Testa a leitura de um arquivo Parquet."""
        # Configura o mock
        expected_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_parquet.return_value = expected_df

        # Cria a instância de DataReaderFile
        reader = DataReaderFile(self.config_parquet)

        # Lê os dados
        result = reader.read_data()

        # Verifica o resultado
        self.assertTrue(result.equals(expected_df))

    def test_invalid_data_type(self):
        """Testa o comportamento com um tipo de leitura inválido."""
        invalid_config = {
            "reading": {
                "reading_mode": "invalid_type",
                "caminho": "C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets",
                "nome_arquivo": "data.invalid"
            }
        }
        reader = DataReaderFile(invalid_config)
        with self.assertRaises(ValueError):
            reader.read_data()


if __name__ == '__main__':
    unittest.main()