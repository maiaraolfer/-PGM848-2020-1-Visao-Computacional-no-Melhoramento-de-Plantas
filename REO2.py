########################################################################################################################
# DATA: 03/08/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# LISTA DE EXERCÍCIOS - ROTEIRO DE ESTUDOS ORIENTADOS 2
# ALUNA: MAIARA OLIVEIRA FERNANDES - 2018260157  (github.com/maiaraolfer)
########################################################################################################################

### BIBLIOTECAS UTILIZADAS ###
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

print("EXERCÍCIO 1 - Selecione uma imagem a ser utilizada no trabalho prático e "
      "realize os seguintes processos utilizando o pacote OPENCV do Python:")
print("a) Apresente a imagem e as informações de número de linhas e colunas, número de canais e número total de pixels")

img = "img.jpg" # Nome do arquivo a ser utilizado na análise
img = cv2.imread(img,1) # Carrega imagem (0 - Binária e Escala de Cinza; 1 - Colorida (BGR))
print(img)
row, col, canal = np.shape(img) #carrega a matriz da imagem; row,col se for binario; row, col, canal se for colorida;
print('Tipo: ',img.dtype) #uint8 (8 bits)
print('Dimensão: ' + str(row) +' x '+ str(col))
print('Largura da imagem: ' + str(col))
print('Altura da imagem: ' + str(row))
print('Número de canais da imagem:' + str(canal))

#pixels
print('O número de pixels é ' + str(row*col)) # numero de linhas * numero de colunas = total de pixels

#Apresentando imagens no matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #transforma de BGR para RGB
'''
plt.figure('Figura 1a')
plt.imshow(img_rgb) # im.show(imagem, cmap = mapa de cor "grey" - escala de cinza)
plt.xticks([]) # Eliminar o eixo X 'xticks recebe lista vazia'
plt.yticks([]) # Eliminar o eixo Y
plt.title("Imagem") #titulo
plt.show() #plt.show para apresentar a imagem
'''
print('_______________________________________________________________________________________________________________')

print("b) Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem"
      " para a solução das próximas alternativas;")
#plotando a figura com os eixos para facilitar o recorte
#plt.figure('Imagem RGB')
#plt.imshow(img_rgb)
#plt.show()
#recorte da terceira folha
folha3 = img_rgb[:2400,2610:3880] #linhas = eixo y "decrescente"; col = eixo x "crescente"
folha3_bgr = cv2.cvtColor(folha3, cv2.COLOR_RGB2BGR) #folha3 = RGB >> para salvar é preciso transformar novamente para BGR
cv2.imwrite('folha3.jpg', folha3) #salvando a imagem recortada - imwrite('nome.extensão',imagem)

'''
plt.figure('Figura 1b')
plt.imshow(folha3)
plt.title("Recorte folha 3")
plt.show()
'''

print('______________________________________________________________________________________________________________')

print("c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando "
      "os mapas de cores escala de cinza e JET")
#carregando a nova imagem
img_folha3 = "folha3.jpg"
img_folha3_cinza = cv2.imread(img_folha3, 0) #folha3 = imagem RGB

'''
plt.figure('Figura 1c')
plt.subplot(1,2,1) #1 linha 2 colunas imagem 1
plt.imshow(img_folha3_cinza, cmap='gray')
plt.title("Escala de cinza")
plt.colorbar(orientation = 'horizontal')
plt.subplot(1,2,2) #imagem 2
plt.imshow(img_folha3_cinza, cmap='jet')
plt.title("Escala JET")
plt.colorbar(orientation = 'horizontal')
plt.show()
print('_______________________________________________________________________________________________________________')
'''
print("d) Apresente a imagem em escala de cinza e o seu respectivo histograma. Relacione o histograma e a imagem.")

row,col = np.shape(img_folha3_cinza)
print('Dimensão: ' + str(row) +' x '+ str(col))
print('O número total de pixels é '+str(row*col))
hist = cv2.calcHist(img_folha3_cinza,[0],None, [256], [0,256])
'''
plt.figure("Figura 1d")
plt.subplot(1,2,1)
plt.imshow(img_folha3_cinza, cmap='gray')
plt.title('Escala de cinza')
plt.colorbar(orientation = 'horizontal')

plt.subplot(1,2,2)
plt.plot(hist, color='black')
plt.title('Histograma')
plt.xlim([0,256])
plt.xlabel("Valores dos pixels")
plt.ylabel("Número de pixels")
plt.show()
print('_______________________________________________________________________________________________________________')
'''
print("e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o fundo"
      " da imagem utilizando um limiar manual e o limiar obtido pela técnica de Otsu. Nesta questão apresente o"
      " histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final"
      " obtida da segmentação. Explique os resultados.")

## HISTOGRAMA EM ESCALA DE CINZA (1d)
#hist = cv2.calcHist(img_folha3_cinza,[0],None, [256], [0,256])

#LIMIARIZAÇÃO MANUAL
# imagem binária >> 0 = preto; 1 = branco (escala de intensidade);
# imagem escala de cinza >> 0 = preto e 255 = branco
limiar_cinza = 150 # valores acima de 140 recebem valor 1; valores abaixo recebem valor 0;
(limiar, img_limiar) = cv2.threshold(img_folha3_cinza,limiar_cinza,255,cv2.THRESH_BINARY) #binária
(limiar, img_limiar_inv) = cv2.threshold(img_folha3_cinza,limiar_cinza,255,cv2.THRESH_BINARY_INV) #binária invertida
# parametros da função: imagem, valor do limiar = 140, valor maximo = 255, threshold binario - imagem binária
# a função cv2.threshold solta o limiar - no caso, por ser manual, ja sabemos qual é.
# e também a imagem binarizada; 'binary_inv' inverte os valores recebidos>> >140 = 0; <140 = 1)
print('Limiar: ' + str(limiar))

## LIMIARIZAÇÃO TECNICA OTSU
(limiar_otsu, img_limiar_otsu) = cv2.threshold(img_folha3_cinza,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#0 = limiar ; valor maximo; tresh_binary + thresh_otsu - a tecnica fornece o limiar mais adequado automaticamente;
'''
#Apresentando as imagens - limiar manual
plt.figure('Figura 1e - Limiar manual')
plt.subplot(2,3,1) #2 linhas 3 colunas imagem 1
plt.imshow(folha3) #imagem folha 3 em RGB
plt.title('RGB')

plt.subplot(2,3,2) #imagem folha 3 em escala de cinza
plt.imshow(img_folha3_cinza,cmap='gray')
plt.title('Escala de cinza')

plt.subplot(2,3,3) #histograma escala de cinza
plt.plot(hist, color='black')
plt.axvline(x=limiar_cinza,color='red') #linha vertical indicando o valor do limiar
plt.title("Histograma - Limiar Manual")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,4) # imagem binarizada - limiarizada; imagem binaria apos usar o limiar de 140
plt.imshow(img_limiar,cmap='gray')
plt.title('Imagem binarizada - L: ' + str(limiar_cinza))

plt.subplot(2,3,5) #imagem binarizada invertida
plt.imshow(img_limiar_inv, cmap = 'gray')
plt.title('Imagem binarizada invertida')
plt.show()

plt.figure('Figura 1e - Limiar OTSU')
plt.subplot(2,2,1) #2 linhas 3 colunas imagem 1
plt.imshow(folha3) #imagem folha 3 em RGB
plt.title('RGB')

plt.subplot(2,2,2) #imagem folha 3 em escala de cinza
plt.imshow(img_folha3_cinza,cmap='gray')
plt.title('Escala de Cinza')

plt.subplot(2,2,3) #histograma escala de cinza
plt.plot(hist, color='black')
plt.axvline(x=limiar_otsu,color='red') #linha vertical indicando o valor do limiar de OTSU
plt.title("Histograma - Limiar OTSU ")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,4) #imagem binarizada - limiarizada; imagem binaria apos usar o limiar de 140
plt.imshow(img_limiar_otsu,cmap='gray')
plt.title('Imagem binarizada - L: ' + str(limiar_otsu))
plt.show()
'''
#SEGMENTAÇÃO
img_seg = cv2.bitwise_and(folha3, folha3, mask=img_limiar_otsu)
#salvando
img_seg_bgr = cv2.cvtColor(img_seg, cv2.COLOR_RGB2BGR)
cv2.imwrite('folha3_seg.png', img_seg_bgr)
'''
plt.figure("Segmentação 1e")
plt.imshow(img_seg)
plt.xticks([])
plt.yticks([])
plt.show()
print('_______________________________________________________________________________________________________________')
'''
print("f) Apresente uma figura contendo a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.")
folha3_Lab = cv2.cvtColor(folha3,cv2.COLOR_RGB2Lab)
folha3_hsv = cv2.cvtColor(folha3, cv2.COLOR_RGB2HSV)
folha3_YCrCb = cv2.cvtColor(folha3, cv2.COLOR_RGB2YCrCb)
'''
plt.figure('Figura 1f')
plt.subplot(1,4,1)
plt.imshow(folha3)
plt.title('RGB')

plt.subplot(1,4,2)
plt.imshow(folha3_Lab)
plt.title('Lab')

plt.subplot(1,4,3)
plt.imshow(folha3_hsv)
plt.title('HSV')

plt.subplot(1,4,4)
plt.imshow(folha3_YCrCb)
plt.title('YCrCb')
plt.show()
print('_______________________________________________________________________________________________________________')
'''
print("g) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de "
      "cada um dos canais e seus respectivos histogramas.")
## RGB
r,g,b = cv2.split(folha3)
hist_r = cv2.calcHist([folha3],[0],None,[256],[0,256])
hist_g = cv2.calcHist([folha3],[1],None,[256],[0,256])
hist_b = cv2.calcHist([folha3],[2],None,[256],[0,256])
'''
plt.figure('Figura 1g - RGB')
plt.subplot(2,3,1) #imagens na linha 1
plt.imshow(folha3[:,:,0], cmap = "gray") #todas as linhas, todas as colunas, somente canal 0 = R
plt.title('Canal R')

plt.subplot(2,3,2)
plt.imshow(folha3[:,:,1], cmap = "gray")
plt.title('Canal G')

plt.subplot(2,3,3)
plt.imshow(folha3[:,:,2], cmap = "gray")
plt.title("Canal B")

plt.subplot(2,3,4) ## histogramas na linha 2
plt.plot(hist_r, color='red')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,5)
plt.plot(hist_g, color='green')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.subplot(2,3,6)
plt.plot(hist_b, color='blue')

plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
'''

#Lab
L,a,b2=cv2.split(folha3_Lab)
hist_L=cv2.calcHist([folha3_Lab],[0],None,[256],[0,256])
hist_a=cv2.calcHist([folha3_Lab],[1],None,[256],[0,256])
hist_b2=cv2.calcHist([folha3_Lab],[2],None,[256],[0,256])
'''
plt.figure('Figura 1g - Lab')
plt.subplot(2,3,1) #imagens na linha 1
plt.imshow(folha3[:,:,0], cmap = "gray") #todas as linhas, todas as colunas, somente canal 0 = R
plt.title('Canal L')

plt.subplot(2,3,2)
plt.imshow(folha3[:,:,1], cmap = "gray")
plt.title('Canal a')

plt.subplot(2,3,3)
plt.imshow(folha3[:,:,2], cmap = "gray")
plt.title("Canal b")

plt.subplot(2,3,4)
plt.plot(hist_L, color='black')
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2, 3, 5)
plt.plot(hist_a, color='black')
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2, 3, 6)
plt.plot(hist_b2, color='black')
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
'''
#HSV
h,s,v = cv2.split(folha3_hsv)
hist_h = cv2.calcHist([folha3_hsv],[0],None, [256], [0,256])
hist_s = cv2.calcHist([folha3_hsv],[1],None, [256], [0,256])
hist_v = cv2.calcHist([folha3_hsv],[2],None, [256], [0,256])
'''
plt.figure('Figura 1g - HSV')
plt.subplot(2,3,1) #imagens na linha 1
plt.imshow(folha3[:,:,0], cmap = "gray") #todas as linhas, todas as colunas, somente canal 0 = R
plt.title('Canal H')

plt.subplot(2,3,2)
plt.imshow(folha3[:,:,1], cmap = "gray")
plt.title('Canal S')

plt.subplot(2,3,3)
plt.imshow(folha3[:,:,2], cmap = "gray")
plt.title("Canal V")

plt.subplot(2,3,4)
plt.plot(hist_h, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,5)
plt.plot(hist_s, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,6)
plt.plot(hist_v, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
'''
#YCrCb
y,cr,cb = cv2.split(folha3_YCrCb)
hist_y = cv2.calcHist([folha3_YCrCb],[0],None, [256], [0,256])
hist_cr = cv2.calcHist([folha3_YCrCb],[1],None, [256], [0,256])
hist_cb = cv2.calcHist([folha3_YCrCb],[2],None, [256], [0,256])
'''
plt.figure('Figura 1g - YCrCb')
plt.subplot(2,3,1) #imagens na linha 1
plt.imshow(folha3[:,:,0], cmap = "gray") #todas as linhas, todas as colunas, somente canal 0 = R
plt.title('Canal Y')

plt.subplot(2,3,2)
plt.imshow(folha3[:,:,1], cmap = "gray")
plt.title('Canal Cr')

plt.subplot(2,3,3)
plt.imshow(folha3[:,:,2], cmap = "gray")
plt.title("Canal Cb")

plt.subplot(2,3,4)
plt.plot(hist_y, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,5)
plt.plot(hist_cr, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,6)
plt.plot(hist_cb, color='black')
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
'''
print('_______________________________________________________________________________________________________________')
print("h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem de modo a remover "
      "o fundo da imagem utilizando limiar manual e limiar obtido pela técnica de Otsu. Nesta questão apresente o "
      "histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final "
      "obtida da segmentação. Explique os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação."
      " Nesta questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.")
#teste dos sistemas de cor com limiar OTSU para encontrar o melhor sistema de cor e canal
'''
#RGB
(L1, img_limiar_otsu_r) = cv2.threshold(folha3[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L2, img_limiar_otsu_g) = cv2.threshold(folha3[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L3, img_limiar_otsu_b) = cv2.threshold(folha3[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#HSV
(L4, img_limiar_otsu_h) = cv2.threshold(folha3_hsv[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L5, img_limiar_otsu_s) = cv2.threshold(folha3_hsv[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L6, img_limiar_otsu_v) = cv2.threshold(folha3[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Lab
(L7, img_limiar_otsu_L) = cv2.threshold(folha3_Lab[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L8, img_limiar_otsu_a) = cv2.threshold(folha3_Lab[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L9, img_limiar_otsu_b2) = cv2.threshold(folha3_Lab[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#YCrCb
(L10, img_limiar_otsu_Y) = cv2.threshold(folha3_YCrCb[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L11, img_limiar_otsu_Cr) = cv2.threshold(folha3_YCrCb[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L12, img_limiar_otsu_Cb) = cv2.threshold(folha3_YCrCb[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure("1h - Sistema RGB")
plt.subplot(1,3,1)
plt.imshow(img_limiar_otsu_r,cmap='gray')
plt.title('Canal R')
plt.subplot(1,3,2)
plt.imshow(img_limiar_otsu_g, cmap='gray')
plt.title('Canal G')
plt.subplot(1,3,3)
plt.imshow(img_limiar_otsu_b, cmap='gray')
plt.title('Canal B')
plt.show()

plt.figure("1h - Sistema HSV")
plt.subplot(1,3,1)
plt.imshow(img_limiar_otsu_h, cmap='gray')
plt.title('Canal H')
plt.subplot(1,3,2)
plt.imshow(img_limiar_otsu_s, cmap='gray')
plt.title('Canal S')
plt.subplot(1,3,3)
plt.imshow(img_limiar_otsu_v, cmap='gray')
plt.title('Canal V')
plt.show()

plt.figure("1h - Sistema Lab")
plt.subplot(1,3,1)
plt.imshow(img_limiar_otsu_L, cmap='gray')
plt.title('Canal L')
plt.subplot(1,3,2)
plt.imshow(img_limiar_otsu_a, cmap='gray')
plt.title('Canal a')
plt.subplot(1,3,3)
plt.imshow(img_limiar_otsu_b2,cmap='gray')
plt.title('Canal b')
plt.show()

plt.figure("1h - Sistema YCrCb")
plt.subplot(1,3,1)
plt.imshow(img_limiar_otsu_Y, cmap='gray')
plt.title('Canal Y')
plt.subplot(1,3,2)
plt.imshow(img_limiar_otsu_Cr, cmap='gray')
plt.title('Canal Cr')
plt.subplot(1,3,3)
plt.imshow(img_limiar_otsu_Cb, cmap='gray')
plt.title('Canal Cb')
plt.show()
'''
## o canal S foi o que melhor discriminou a folha do fundo;
limiar= 70
(limiar, img_limiar_manual_s) = cv2.threshold(folha3_hsv[:,:,1],limiar,255,cv2.THRESH_BINARY)
(L5, img_limiar_otsu_s) = cv2.threshold(folha3_hsv[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#segmentação
folha3_seg_manual=cv2.bitwise_and(folha3, folha3, mask = img_limiar_manual_s)
folha3_seg_otsu=cv2.bitwise_and(folha3, folha3, mask = img_limiar_otsu_s)

#apresentação
'''
plt.figure("Figura 1h_manual") #images limiarizadas
plt.subplot(2,2,1)
plt.imshow(img_limiar_manual_s, cmap = "gray")
plt.xticks([])
plt.yticks([])
plt.title('Segmentação manual')

plt.subplot(2,2,3) #histogramas do canal S
plt.plot(hist_s, color = "black")
plt.title("Limiar manual: 70")
plt.axvline(x = limiar, color = "red")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,2) #imagens segmentadas
plt.imshow(folha3_seg_manual)
plt.show()

plt.figure('Figura 1h_otsu')
plt.subplot(2,2,1)
plt.imshow(img_limiar_otsu_s, cmap = "gray")
plt.xticks([])
plt.yticks([])
plt.title("Segmentação Otsu")

plt.subplot(2,2,3)
plt.plot(hist_s, color = "black")
plt.title("Limiar Otsu:" + str(L5))
plt.axvline(x=L5, color = "red")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,2)
plt.imshow(folha3_seg_otsu)
plt.show()
'''
print('_______________________________________________________________________________________________________________')
print("i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada "
      "(binarizada) da letra h.")

hist_lim_r = cv2.calcHist([folha3], [0], img_limiar_otsu_s, [256], [0,256])
hist_lim_g = cv2.calcHist([folha3], [1], img_limiar_otsu_s, [256], [0,256])
hist_lim_b = cv2.calcHist([folha3], [2], img_limiar_otsu_s, [256], [0,256])
'''
plt.figure("Figura 1i")
plt.subplot(3,1,1)
plt.plot(hist_lim_r, color = "r")
plt.title("Canal R")
plt.xlim([0,256])
#plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,1,2)
plt.plot(hist_lim_g, color = "g")
plt.title("Canal G")
plt.xlim([0,256])
#plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,1,3)
plt.plot(hist_lim_b, color = "b")
plt.title("Canal B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
'''
print('_______________________________________________________________________________________________________________')
print("j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse. "
      "Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas.")

operacao = ((2*folha3_seg_otsu[:,:,0]) - (0.5 *folha3_seg_otsu[:,:,1]))
operacao = operacao.astype(np.uint8)

hist = cv2.calcHist([operacao],[0],None,[256],[1,255])
limiar,ferrugem = cv2.threshold(operacao,180,255,cv2.THRESH_BINARY)

'''
plt.figure('Figura 1j')
plt.subplot(1,3,1)
plt.imshow(operacao,cmap='jet')
plt.subplot(1,3,2)
plt.imshow(folha3)
plt.subplot(1,3,3)
plt.imshow(ferrugem,cmap='jet')
plt.show()
'''









