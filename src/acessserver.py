import polars as pl
import paramiko
from io import BytesIO
from typing import Optional, Generator

class AcessServer:
    def __init__(self, host: str, username: str, password: str, remote_path: str):
        """
        Inicializa a classe com o JSON de configuração.

        :param config_json: String contendo o JSON de configuração.
        """
        self.host = host
        self.username = username
        self.password = password
        self.remote_path = remote_path

    def get_file_chunks(self, chunk_size: int = 1024 * 1024) -> Generator[bytes, None, None]:
        """
        Lê um arquivo remoto em blocos (chunks) sem carregar todo o conteúdo na memória.

        :param chunk_size: Tamanho de cada bloco (padrão: 1 MB).
        :return: Generator que produz blocos de bytes.
        """
        # Conecta ao servidor remoto via SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Iniciando conexão ao servidor remoto: {self.host}")

        try:
            ssh.connect(self.host, username=self.username, password=self.password)
        except Exception as e:
            raise ValueError(f"Não foi possivel se conectar ao servidor. Erro: {e}")
            exit()

        # Abre um canal SFTP para transferir o arquivo
        sftp = ssh.open_sftp()

        try:
            with sftp.file(self.remote_path, 'rb') as remote_file:
                if chunk_size is not None:
                    while True:
                        chunk = remote_file.read(chunk_size)  # Lê um bloco de dados
                        if not chunk:
                            break  # Sai do loop quando não há mais dados
                        yield chunk
                else:
                    chunk = remote_file.read()
                    yield chunk
        finally:
            # Fecha a conexão SFTP e SSH
            sftp.close()
            ssh.close()