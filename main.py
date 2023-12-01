import cv2
from ultralytics import YOLO
import threading

camera = cv2.VideoCapture(0)  # Inicializar a câmera

modelo = YOLO('yolov8n.pt')  # Carregar o modelo YOLO

area = [100, 190, 1150, 700]  # Definir a área de interesse

while True:
    ret, img = camera.read()  # Ler a imagem da câmera
    img = cv2.resize(img, (1270, 720))  # Redimensionar a imagem
    img2 = img.copy()  # Copiar a imagem
    cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), -1)
    resultado = modelo(img)  # Executar o modelo YOLO

    for objetos in resultado:  # Percorrer os objetos detectados
        obj = objetos.boxes  # Obter os objetos detectados
        for dados in obj:  # Percorrer os dados dos objetos detectados
            x, y, w, h = dados.xyxy[0]  # Obter as coordenadas do objeto
            x, y, w, h = int(x), int(y), int(w), int(h)  # Converter as coordenadas para inteiros
            cls = int(dados.cls[0])  # Obter a classe do objeto
            cx, cy = (x + w) // 2, (y + h) // 2  # Obter o centro do objeto
            if cls == 0:  # Se a classe for 0 (pessoa)
                cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 5)  # Desenhar um retângulo azul
                # Verificar se o centro do objeto está dentro da área de interesse
                if area[0] <= cx <= area[2] and area[1] <= cy <= area[3]:
                    cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
                    cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
                    cv2.putText(img, 'INVASOR DETECTADO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)  # Mesclar as duas imagens

    cv2.imshow('Video', imgFinal)  # Exibir a imagem

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Se pressionar a tecla Q
        break

# Liberar os recursos da câmera
camera.release()
cv2.destroyAllWindows()
