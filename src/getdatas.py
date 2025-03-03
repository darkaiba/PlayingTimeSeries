import polars as pl
from typing import Dict, Any
from src.acessserver import AcessServer
from typing import Optional, Generator
from io import BytesIO
from src import TypeFile

class DataReaderRemote:
    def __init__(self, config_json):
        """
        Inicializa a classe com o JSON de configuração.

        :param config_json: String contendo o JSON de configuração.
        """
        self.config = config_json['reading']

    def read_data(self, chunk_size: int):
        """
        Lê os dados com base no tipo de leitura especificado no JSON.
        """
        data_type = str(self.config['reading_mode']).lower()

        print("Lendo os dados de entrada")
        if data_type == TypeFile.CSV:
            return self._read_csv(chunk_size)
        elif data_type == TypeFile.DATABASE:
            return self._read_database(chunk_size)
        else:
            raise ValueError(f"Tipo de leitura não suportado: {data_type}")
            exit()

    def _read_csv(self, chunk_size: Optional[int]) -> Generator[pl.DataFrame, None, None]:
        """
        Lê um arquivo CSV.
        """
        host = self.config['host']
        user = self.config['user']
        password = self.config['password']
        path = self.config['path']
        filename = self.config['filename']

        full_path = f"{path}/{filename}"
        acess_server = AcessServer(host, user, password, full_path)

        # Lê o arquivo em blocos
        print("Iniciando a leitura do arquivo CSV em blocos")
        buffer = b""
        for chunk in acess_server.get_file_chunks(chunk_size=1024 * 1024):  # 1 MB por bloco
            buffer += chunk
            lines = buffer.split(b'\n')
            buffer = lines.pop()  # Guarda a linha incompleta para o próximo bloco

            # Processa as linhas completas
            if lines:
                file_like = BytesIO(b'\n'.join(lines))
                df = pl.read_csv(file_like, try_parse_dates=True)
                yield df

        # Processa o último bloco (se houver)
        if buffer:
            file_like = BytesIO(buffer)
            df = pl.read_csv(file_like)
            yield df

    def _read_database(self, chunk_size: Optional[int]) -> Generator[pl.DataFrame, None, None]:
        """
        Lê dados de um banco de dados.
        """
        MYSQL = 'mysql'
        ATHENA = 'athena'
        REDSHIFT = 'redshift'
        POSTGRES = 'postgres'
        SQLSERVER = 'sqlserver'

        host = self.config['host']
        user = self.config['user']
        password = self.config['password']
        database = self.config['database']
        type_database = self.config['type_database']

        if str(type_database).lower() == MYSQL:
            type_database = "mysql+pymysql" #ou mysql
        elif str(type_database).lower() == ATHENA:
            type_database = "awsathena+rest" #ou "awsathena"
        elif str(type_database).lower() == REDSHIFT:
            type_database = "redshift+psycopg2" #ou "redshift"
        elif str(type_database).lower() == POSTGRES:
            type_database = "postgresql+psycopg2" #ou "postgresql"
        elif str(type_database).lower() == SQLSERVER:
            type_database = "mssql+pyodbc" #ou "mssql"
        else:
            raise ValueError(f"Tipo de banco de dados não reconhecido: {str(type_database).lower()}")
            exit()

        print(f"Iniciando o acesso ao banco de dados: {host}/{database}")
        try:
            connection_string = f"{type_database}://{user}:{password}@{host}/{database}"
        except Exception as e:
            raise ValueError(f"Nao foi possivel se conectar ao banco de dados. Erro: {e}")
            exit()

        try:
            if chunk_size:
                # Implementação de leitura em chunks para banco de dados
                offset = 0
                while True:
                    chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
                    df = pl.read_database(chunk_query, connection_string)
                    if df.is_empty():
                        break
                    yield df
                    offset += chunk_size
            else:
                # Lê todos os dados de uma vez
                yield pl.read_database(query, connection)
        finally:
            # Fecha a conexão com o banco de dados
            connection.close()
            pass

class DataReaderFile:
    def __init__(self, config_json):
        """
        Inicializa a classe com o JSON de configuração.

        :param config_json: String contendo o JSON de configuração.
        """
        self.config = config_json['reading']

    def read_data(self):
        """
        Lê os dados com base no tipo de leitura especificado no JSON.
        """
        data_type = str(self.config['reading_mode']).lower()

        print("Lendo os dados de entrada")
        if data_type == TypeFile.CSV:
            return self._read_csv()
        elif data_type == TypeFile.JSON:
            return self._read_json()
        elif data_type == TypeFile.PARQUET:
            return self._read_parquet()
        else:
            raise ValueError(f"Tipo de leitura não suportado: {data_type}")
            exit()

    def _read_csv(self) -> Generator[pl.DataFrame, None, None]:
        """
        Lê um arquivo CSV.
        """
        path = self.config['path']
        filename = self.config['filename']

        full_path = f"{path}\\{filename}"

        # Lê o arquivo
        print(f"Iniciando a leitura do arquivo CSV")
        return pl.read_csv(full_path, truncate_ragged_lines=True, try_parse_dates=True)

    def _read_json(self) -> Generator[pl.DataFrame, None, None]:
        """
        Lê um arquivo JSON.
        """
        path = self.config['path']
        filename = self.config['filename']

        full_path = f"{path}/{filename}"

        # Lê o arquivo em chunks
        print("Iniciando a leitura do arquivo JSON")
        return pl.read_json(full_path)

    def _read_parquet(self) -> Generator[pl.DataFrame, None, None]:
        """
        Lê um arquivo Parquet.
        """
        path = self.config['path']
        filename = self.config['filename']

        full_path = f"{path}/{filename}"

        # Se chunk_size for fornecido, divide a leitura em partes
        print("Iniciando a leitura do arquivo PARQUET")
        return pl.read_parquet(full_path)

