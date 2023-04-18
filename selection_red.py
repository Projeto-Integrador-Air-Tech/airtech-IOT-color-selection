import cv2
import numpy as np

# define a cor a ser detectada (vermelha)
color_lower = np.array([0, 50, 50])
color_upper = np.array([10, 255, 255])

# inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # captura um frame do vídeo
    ret, frame = cap.read()
    
    # converte a imagem para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # aplica a filtragem da cor
    mask = cv2.inRange(hsv_frame, color_lower, color_upper)
    
    # aplica a operação de abertura e fechamento na máscara
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # encontra as regiões com a cor
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.rectangle(frame, (int(x - radius), int(y - radius)), (int(x + radius), int(y + radius)), (0, 255, 0), 2)
            cv2.putText(frame, "Vermelho", (int(x - radius), int(y - radius - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Cor detectada: Vermelho")
    else:
        cv2.imshow("Video", frame)
    
    # exibe o frame com o resultado da detecção
    cv2.imshow("Video", frame)
    
    # sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) == ord('q'):
        break

# libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
