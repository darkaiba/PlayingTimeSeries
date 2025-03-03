from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class Report:

    def __init__(self):
        return

    def create_report(self, name, name_images, path_job, metrics):

        # Criando um PDF
        c = canvas.Canvas(f"{path_job}/relatorio/Relatorio_{name}.pdf", pagesize=letter)

        c.setFont("Helvetica", 14)

        # Posição inicial para desenhar as imagens
        x = 100  # Posição horizontal
        y = 750  # Posição vertical (começa no topo da página)

        # Adicionando imagens
        for imagen in name_images:
            c.drawImage(f"{path_job}/imagem/{imagen}", x, y-200, width=400, height=200)

            # Atualiza a posição y para o próximo gráfico
            y -= 250  # Espaçamento entre os gráficos

            # Verifica se é necessário criar uma nova página
            if y < 100:  # Se a posição y estiver muito baixa
                c.showPage()  # Cria uma nova página
                y = 750  # Reinicia a posição y no topo da nova página

        for metric in metrics:

            # Adiciona o texto descritivo abaixo da imagem
            text_object = c.beginText(x, y-20)  # Posiciona o texto abaixo da imagem
            text_object.setFont("Helvetica", 12)
            text_object.textLines(metric)  # Quebra o texto em várias linhas
            c.drawText(text_object)

            # Atualiza a posição y para o próximo gráfico
            y -= 50  # Espaçamento entre os gráficos

            # Verifica se é necessário criar uma nova página
            if y < 100:  # Se a posição y estiver muito baixa
                c.showPage()  # Cria uma nova página
                y = 750  # Reinicia a posição y no topo da nova página

        return c

    def save_report(self, canva):
        # Finalizando o PDF
        canva.save()