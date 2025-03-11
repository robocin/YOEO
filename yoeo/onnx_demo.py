import onnxruntime as ort
import cv2
import os
import numpy as np
from yoeo.utils.class_config import ClassConfig
from yoeo.utils.dataclasses import ClassNames, GroupConfig

modelPath = "/home/drones/Documents/YOEO/weights/yoeo.onnx"
imagesPath = "/home/drones/Documents/YOEO/data/samples"
model = ort.InferenceSession(modelPath)

detection_classes = ['ball', 'goalpost', 'robot']
segmentation_classes = ['background', 'lines', 'field']

def draw_detections(image, detections, class_names):
    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
        
        # Convertendo para inteiros para desenhar
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Desenhando a caixa de detecção
        color = (0, 255, 0)  # Cor da caixa (verde, pode ser alterado)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Adicionando o rótulo da classe
        label = f"{class_names[int(class_id)]}: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Função para desenhar as segmentações
def draw_translucent_segmentations(image, segmentations, class_names):
    # Para cada classe de segmentação, cria uma máscara binária
    overlay = image.copy()  # Cópia da imagem original
    alpha = 0.5  # Nível de transparência (0 é totalmente transparente, 1 é totalmente opaco)
    
    for i in range(len(class_names)):
        mask = segmentations == i  # Máscara binária para a classe i
        
        if i == 1:  # Exemplo: 'lines' é colorido em vermelho
            color = (0, 0, 255)  # Vermelho
        elif i == 2:  # Exemplo: 'field' é colorido em azul
            color = (255, 0, 0)  # Azul
        else:
            continue  # Para o fundo, não faz nada

        # Aplica a cor na área da máscara
        overlay[mask] = color
        
    # Fazendo a sobreposição translúcida (mistura da imagem original com a máscara colorida)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image

# Convert to numpy
for element in os.listdir(imagesPath):
    imgPath = os.path.join(imagesPath, element)
    img = cv2.imread(imgPath)
    original_height, original_width = img.shape[:2]

    # Redimensionando a imagem para 416x416
    img_resized = cv2.resize(img, (416, 416))
    
    data = np.array(img_resized, dtype=np.float32)
    data = np.transpose(data, (2, 0, 1))  # Convertendo para (C, H, W)
    data = np.expand_dims(data, axis=0)  # Adicionando o batch dimension

    outputs = model.run(None, {"InputLayer": data})
    
    # Convert the output to detections --> tensor: float32[1,2535,8] 
    # Segmentations --> tensor: uint8[1,416,416]
    detections = outputs[0]
    segmentations = outputs[1]

    # Remove the batch dimension from the segmentations
    segmentations = np.squeeze(segmentations, axis=0)

    # Interpretando as detecções
    detected_boxes = detections[0]  # (2535, 8)
    detected_boxes = detected_boxes[detected_boxes[:, 4] > 0.9]  # Filtrando detecções com confiança > 0.9

    # Desenhando detecções na imagem redimensionada
    img_with_detections = draw_detections(img_resized.copy(), detected_boxes, detection_classes)

    # Aplicando a segmentação translúcida na imagem redimensionada
    img_with_segmentations = draw_translucent_segmentations(img_resized.copy(), segmentations, segmentation_classes)

    # Redimensionando as imagens para as dimensões originais antes de exibir
    img_with_detections_resized = cv2.resize(img_with_detections, (original_width, original_height))
    img_with_segmentations_resized = cv2.resize(img_with_segmentations, (original_width, original_height))

    # Exibindo a imagem com detecções
    cv2.imshow("Image with Detections", img_with_detections_resized)

    # Exibindo a imagem com segmentações translúcidas
    cv2.imshow("Image with Translucent Segmentations", img_with_segmentations_resized)

    cv2.waitKey(0)  # Aguarda até que uma tecla seja pressionada para fechar as janelas
    cv2.destroyAllWindows()


import onnxruntime as ort
import cv2
import os
import numpy as np
from yoeo.utils.class_config import ClassConfig
from yoeo.utils.dataclasses import ClassNames, GroupConfig


modelPath = "/home/drones/Documents/YOEO/weights/yoeo.onnx"
imagesPath = "/home/drones/Documents/YOEO/data/samples"
model = ort.InferenceSession(modelPath)

#Convert to numpy
for element in os.listdir(imagesPath):
    imgPath = os.path.join(imagesPath, element)
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (416, 416))
    data = np.array(img)
    #print(data.shape)
    
    #Convert to tensor: float32[1,3,416,416]
    data = np.array(data, dtype=np.float32)
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    #print(data.shape)


    outputs = model.run(None, {"InputLayer": data})
    
    #Convert the output to detections --> tensor: float32[1,2535,8] 
    #Segmentations --> tensor: uint8[1,416,416]
    detections = outputs[0]
    segmentations = outputs[1]
    segmentations = np.squeeze(segmentations, axis=0)

    #printing the output
    detections = np.squeeze(detections, axis=0)

    print(detections)
    print("number of detections: ", len(detections))