import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from deepface import DeepFace
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

caminho_imagens = r"C:\Users\rosalinda.groner\Documents\GitHub\analise-de-expressoes-deepface\deepface"

if os.path.exists(caminho_imagens):
    
    arquivos = os.listdir(caminho_imagens)
    
    imagens = [arq for arq in arquivos if arq.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not imagens:
        print("Nenhuma imagem encontrada no diretório!")
    else:
        for imagem_nome in imagens:
            caminho_completo = os.path.join(caminho_imagens, imagem_nome)
            
            imagem = cv2.imread(caminho_completo)
            if imagem is not None:
                try:
                    resultado = DeepFace.analyze(imagem, actions=("emotion",), enforce_detection=False)
                    print("")
                    print(imagem_nome)
                    print(f"Emoção dominante: {resultado[0]['dominant_emotion']}")
                    print("")
                except Exception as e:
                    print(f"Erro: {e}")
            else:
                print("Falha ao carregar imagem")
else:
    print(f"Diretório não existe: {caminho_imagens}")