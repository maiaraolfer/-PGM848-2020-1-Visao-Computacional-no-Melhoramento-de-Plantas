########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# ALUNA: MAIARA OLIVEIRA FERNANDES - 2018260157  (github.com/maiaraolfer)
########################################################################################################################

# REO 01 - LISTA DE EXERCÍCIOS
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import math
print('EXERCÍCIO 01')
print('a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.')
x = np.array([43.5, 150.30, 17, 28, 35, 79, 20, 99.07, 15])
print('Vetor x: '+str(x))
print(type(x))
print('_______________________________________________________________________________________________________________')

print('b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.')
dim1 = len(x)
print(type(dim1))
print('O tamanho do vetor x é: ' + str(dim1))
dim2 = np.shape(x)
print('O tamanho do vetor x é: ' + str(dim2))
print(type(dim2))
media = np.mean(x)
print('A média do vetor x é: ' + str(media))
maximo = max(x)
print('O maior valor dentre os elementos do vetor x é: ' + str(maximo))
minimo = min(x)
print('O menor valor dentre os elementos do vetor x é: ' + str(minimo))
sigma = np.var(x)
print('A variância populacional do vetor x é: ' + str(sigma))
print('_______________________________________________________________________________________________________________')

print('c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença'
      'entre cada elemento do vetor declarado na letra A e o valor da média deste')
x2 = (x - media)**2
print('O novo vetor obtido é: ' + str(x2))
print('_______________________________________________________________________________________________________________')

print('d) Obtenha um novo vetor que contenha todos os valores superiores a 30.')
print('Vetor x: '+ str(x))
x3_bool = x>30
x3 = x[x3_bool]
print('O novo vetor que contém apenas valores maiores que 30 é: ' + str(x3))
print('_______________________________________________________________________________________________________________')

print('e) Identifique quais as posições do vetor original possuem valores superiores a 30')
print('Vetor x: '+str(x))
x4 = np.where(x>30)
print('As posições do vetor que possuem valores maiores que 30: ' + str(x4[0]))
#saida é uma lista; a posicao 0 tem as posicoes com valores superiores a 30
print('_______________________________________________________________________________________________________________')

print('f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.')
print('Vetor x: '+str(x))
x5_0 = x[0] #acessa o elemento da posição 0 (primeira posicao)
x5_4 = x[4] #acessa o elemento da posicao 5 (quinta posicao)
x5_8 = x[8] #acessa o elemento da posicao 8 (ultima posicao)
x5 = [x5_0, x5_4, x5_8] #novo vetor com as posicoes que acessamos
print('O vetor com valores da primeira, quinta e ultima posição é: ' +str(x5))
print('_______________________________________________________________________________________________________________')

print('g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua '
      'respectiva posição durante as iterações.')
print('Vetor x: '+str(x))
print('Número de elementos de x: ', len(x))
it = 0
for pos, valor in enumerate(x):
      it = it + 1
      print('Iteração '+ str(it))
      print('Na posição ' + str(pos) + ' temos o valor ' + str(valor))
      time.sleep(0.5)
      print('_________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________')

print('h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.')
print('Vetor x: '+ str(x))
it = 0
sum = 0
for i in x:
    sq = i**2
    sum = sum + sq
    it = it + 1
    print('Iteração: ' + str(it))
    print('Elemento: ' + str(i) + ', o quadrado é: ' + str(sq))
    print('A soma de quadrados é ' + str(sum))
    time.sleep(0.5)
    print('___________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________')

print('i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor.')
print('Vetor x: ' + str(x))
print('Dimensão de x: ' + str(len(x)))
count = 0
while count != 9:
    count +=1
    print('Contador',count)
    time.sleep(0.5)
    print('___________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________')

print('j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.')
print('Dimensão do vetor x: ', str(len(x)))
x6 = range(1, 10, 1) # como começa no 1, para ter 9 elementos, deve ir até o elemento 10.
x6_ = np.array(x6)
print('Sequencia: ' + str(x6_))
print('Dimensão x6: ' + str(len(x6_)))
print('_______________________________________________________________________________________________________________')

print('k) Concatene o vetor da letra a com o vetor da letra j.')
x7 = np.concatenate((x, x6_))
print('O novo vetor é: '+ str(x7))
print('_______________________________________________________________________________________________________________')

########################################################################################################################
########################################################################################################################

print('EXERCÍCIO 02')
print('a) Declare a matriz abaixo com a biblioteca numpy.')
matriz = np.array([[1,3,22],
                   [2,8,18],
                   [3,4,22],
                   [4,1,23],
                   [5,2,52],
                   [6,2,18],
                   [7,2,25]])
print(matriz)
print('_______________________________________________________________________________________________________________')

print('b) Obtenha o número de linhas e de colunas desta matriz.')
nrow,ncol = np.shape(matriz)
print('Nº linhas: ' + str(nrow))
print('Nº colunas: ' + str(ncol))
print('_______________________________________________________________________________________________________________')

print('c) Obtenha as médias das colunas 2 e 3.')
matriz2 = matriz[:,1:] #submatriz com as colunas 2 e 3 da matriz original
print(matriz2)
media2 = np.mean(matriz2, axis=0) #axis = 0 >> colunas
print('Media da matriz: '+str(media2))
print('_______________________________________________________________________________________________________________')

print('d) Obtenha as médias das linhas considerando somente as colunas 2 e 3.')
media3 = np.mean(matriz2, axis = 1) #axis = 1 >> linhas
print('Media das linhas considerando as colunas 2 e 3: '+ str(media3))
print('_______________________________________________________________________________________________________________')

print('e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma '
      'doença e a terceira peso de 100 grãos, obtenha os genótipos que possuem nota de severidade inferior a 5.')
#nota de severidade = segunda coluna - coluna posicao 1; fazer a media de linhas usando a coluna 2 (pos1)
print('Matriz original: ', matriz)
col_notas = matriz[:,1] # pega somente a segunda coluna, das notas.
print('Vetor das notas: ', col_notas)
bool = np.where(col_notas<5) #vetor booleano - retorna T or F, com as posições que atendem a condição dada
print('Posições dos genótipos com notas de severidade menores que 5: ', str(bool[0]))
notas_menor_5 = matriz[:,0][bool]
print('Genótipos com notas de severidades menores que 5: ', str(notas_menor_5))
print('_______________________________________________________________________________________________________________')

print('f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma'
      ' doença e a terceira peso de 100 grãos, obtenha os genótipos que possuem peso de 100 grãos superior'
      ' ou igual a 22.')
# peso de grãos - coluna 3 - posicao 2; obter os genotipos com peso superior ou igual a 22.
print('Matriz original: ', matriz)
col_peso = matriz[:,2]
print('Vetor de peso de 100 grãos: ', col_peso)
bool2 = np.where(col_peso >= 22)
print('Posições dos genótipos com peso de 100 grãos maior ou igual a 22: ', str(bool[0]))
peso_maior_22 = matriz[:,0][bool]
print('Genótipos com peso de 100 grãos maior ou igual a 22: ', str(peso_maior_22))
print('_______________________________________________________________________________________________________________')

print('g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma '
      'doença e a terceira peso de 100 grãos, obtenha os genótipos que possuem nota de severidade igual ou inferior a'
      ' 3 e peso de 100 grãos igual ou superior a 22.')

print('Matriz original: ', matriz)
notas3_peso22 = matriz[(matriz[:,1]<=3) & (matriz[:,2]>=22)]
gen = notas3_peso22[:,0]
print('O genótipos que possuem nota de severidade <= a 3 e peso de 100 graos >= a 22 são: ', str(gen))
print('_______________________________________________________________________________________________________________')
## resposta: genotipos 1,4,5 e 7

print('h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da'
      ' matriz e o seu respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.'
      'Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z". Nesta estrutura,'
      ' crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25.')
matriz = np.array([[1,3,22],
                   [2,8,18],
                   [3,4,22],
                   [4,1,23],
                   [5,2,52],
                   [6,2,18],
                   [7,2,25]]) # resgatando a matriz; 7 linhas e 3 colunas;
nrow,ncol = np.shape(matriz)
print('Matriz', str(matriz))
print('Numero de linhas é '+ str(nrow))
print('Numero de colunas é '+ str(ncol))
print('______________________________________________________________')
## para essa estrutura é preciso e dois loops, um para acessar a posição das linhas e outro para a posição das colunas;
# de forma a acessar um elemento por vez;
count = 0
matriz_zero = np.zeros((nrow, ncol)) # np.zeros(shape, dtype=float, order='c'); dentro do shape tem (nrow,ncol);
for i in np.arange(0, nrow, 1): #for pra acessar as linhas
    for j in np.arange(0, ncol, 1): #for pra acessar as colunas
        count = count + 1
        print('Iteração: ' + str(count))
        print('Na linha ' +str(i) + ' e coluna ' + str(j) + ' ocorre o valor: ' + str(matriz[int(i), int(j)]))
        time.sleep(0.5)
        matriz_zero = (matriz[:,2] >= 25) #peso = terceira coluna (pos2); saída = bool;
        matriz_zero = (matriz[matriz_zero]) # atribui a matriz_zero (que atendem a condição acima) os elementos da matriz original ;
        print('_______________________________________________________________________________________________________')
print('Os genótipos com peso de 100 grãos igual ou superior a 25 são: ' + str(matriz_zero[:,0]))
print('_______________________________________________________________________________________________________________')

########################################################################################################################
########################################################################################################################

print('EXERCÍCIO 03')
print('a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e '
      'a variância amostral um vetor qualquer, baseada em um loop (for).')
## arquivo REO1_3a.py
print('_______________________________________________________________________________________________________________')

print('b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal '
      'com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.')
mu = 100
sigma = math.sqrt(2500)
array1 = np.random.normal(mu, sigma, 10)
print('Simulação 1: ' +str(array1))
array2 = np.random.normal(mu, sigma, 100)
print('Simulação 2: ' + str(array2))
array3 = np.random.normal(mu, sigma, 1000)
print('Simulação 3: ' + str(array3))
print('_______________________________________________________________________________________________________________')

print('c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.')

print('FUNÇÃO PARA CALCULO DA MEDIA')
from REO1_3a_Maiara import media
print('O vetor obtido na simulação 1 é ' + str(array1))
media4 = media(array1)
print('A média do vetor obtido na simulação 1 é ' + str(media4))
print('_______________________________________________________________________________________________________________')
media5 = media(array2)
print('A média do vetor obtido na simulação 2 é ' + str(media5))
print('_______________________________________________________________________________________________________________')
media6 = media(array3)
print('A média do vetor obtido na simulação 3 é ' + str(media6))
print('_______________________________________________________________________________________________________________')

print('FUNÇÃO PARA CALCULO DE VARIÂNCIA')
from REO1_3a_Maiara import variance
print('O vetor obtido na simulação 1 é: ' + str(array1))
var_array1 = variance(array1)
print('A variância do vetor 1 é ' + str(var_array1))
var_array2 = variance(array2)
print('A variância do vetor 2 é ' + str(var_array2))
var_array3 = variance(array3)
print('A variância do vetor 3 é ' + str(var_array3))
print('_______________________________________________________________________________________________________________')

print('d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.')
array4 = np.random.normal(mu, sigma, 10000)
from REO1_3a_Maiara import media
from REO1_3a_Maiara import variance
media7 = media(array4)
var_array4 = variance(array4)

fig, axes = plt.subplots(2, 2) #nrow=2, ncol=2 >> total = 4 graficos
ax0, ax1, ax2, ax3 = axes.flatten() #ordem para posicionar os graficos;
# sem necessidade de usar o plt.subplot() para cada grafico; default = T - linhas; 'F' - colunas;
ax0.hist(array1, 10, density=1, color='midnightblue')
plt.xlabel('Valores')
plt.ylabel('Frequência')
ax0.set_title('Histograma, $\mu=100$, $\sigma=50$, n = 10',fontsize=10)

ax1.hist(array2, 50, density=1, color='navy')
plt.xlabel('Valores')
plt.ylabel('Frequência')
ax1.set_title('n = 100', fontsize=10)
ax2.hist(array3, 100, density=1, color='darkblue')

plt.xlabel('Valores')
plt.ylabel('Frequência')
ax2.set_title('n = 1000',fontsize=10)
ax3.hist(array4, 500, density=1, color='mediumblue')

plt.xlabel('Valores')
plt.ylabel('Frequencia')
ax3.set_title('n = 10000', fontsize=10)
plt.show() #visualização do grafico;

#Salvando o grafico
nome = 'histograma'
fig.savefig((nome+'.png'), bbox_inches="tight")
os.startfile(nome+'.png')
print('_______________________________________________________________________________________________________________')

########################################################################################################################
########################################################################################################################

print('EXERCÍCIO 04')
print('a) O arquivo dados.txt contém a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto '
      'a cinco variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, '
      'apresente os dados e obtenha as informações de dimensão desta matriz.')

dados = np.loadtxt('dados.txt') #carregando os dados
print(dados) #apresentando os dados
nrow, ncol = dados.shape
print('O arquivos dados contém ' + str(nrow) + ' linhas e ' + str(ncol) + ' colunas.')
print('_______________________________________________________________________________________________________________')

print('b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy')
#para pesquisar sobre a utilização de funções é possível usar o comando 'help'
help(np.unique) #encontra um elemento unico dentro de um vetor/matriz

help(np.where) #retorna os elementos de um vetor/matriz dependendo da condição estabelecida
print('_______________________________________________________________________________________________________________')

print('c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas.')
#genotipos = primeira coluna; repetições = segunda coluna;
print('Os genotipos avaliados foram: ' + str(np.unique(dados[:,0])) + 
      ' e o número de repetiçoes avaliadas foi '+ str(max(dados[:,1])))
print('_______________________________________________________________________________________________________________')

print('d) Apresente uma matriz contendo somente as colunas 1, 2 e 4')
submatriz = dados[:,(0,1,3)] #posições das colunas: 0,1 e 3
print(submatriz)
print('_______________________________________________________________________________________________________________')

print('e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da'
      ' coluna 4. Salve esta matriz em bloco de notas.')
matriz_zeros = np.zeros((10,5)) # criando uma matriz em branco, com nrow da submatriz e ncol= 5 (gen, max,min, mean e var)
nrow,ncol = np.shape(matriz_zeros)
print('Linhas: ',nrow) # 10 = 10gen
print('Colunas: ', ncol) # 5 colunas
count = 0
for i in np.arange(0, len(np.unique(submatriz[:,0])), 1):
    count = count + 1
    max = np.max((submatriz[submatriz[:,0] == i+1])[:,-1])
    min = np.min((submatriz[submatriz[:,0] == i+1])[:,-1])
    mean = np.mean((submatriz[submatriz[:,0] == i+1])[:,-1])
    var = np.var((submatriz[submatriz[:,0] == i+1])[:,-1])
    print('Iteração ',count)
    print('Para o genótipo: ' + str(count) + ' o valor máximo da variável resposta é ' + str(max) +
    ' e o valor mínimo é '+ str(min)+ '. A média é ' + str(mean) + ' e a variância é ' + str(var))
    time.sleep(0.5)
    matriz_zeros[i,0] = i+1 #coluna 0 = genotipos;
    matriz_zeros[i,1] = np.max((submatriz[submatriz[:,0] == i+1])[:,-1])
    matriz_zeros[i,2] = np.min((submatriz[submatriz[:,0] == i+1])[:,-1])
    matriz_zeros[i,3] = np.mean((submatriz[submatriz[:,0] == i+1])[:,-1])
    matriz_zeros[i,4] = np.var((submatriz[submatriz[:,0] == i+1])[:,-1])
print(matriz_zeros)

#salvando como .txt
np.savetxt('resultados4e.txt',matriz_zeros, '%2.2f') ## %2 = give the float 2 columns total, 
#and 2f = display 2 positions after the radix point"
print('_______________________________________________________________________________________________________________')

print('f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz '
      'gerada na letra anterior.')
bool = np.where(matriz_zeros[:,3] >= 500) #vetor booleano - retorna T or F, com as posições que atendem a condição dada
print('Posições dos genótipos com média maior que 500: ', str(bool[0]))
media_maior_500 = matriz_zeros[:,0][bool]
print('Genótipos com média maior que 500 são: ', str(media_maior_500))
print('_______________________________________________________________________________________________________________')

print('g) Apresente os seguintes graficos:')
print('1. Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de'
      ' um grafico por figura')
matrix = np.zeros((10,6)) # criando uma matriz em branco, com 10 linhas (1/genotipo) e ncol= 6 (gen + 5 variaveis resposta)
count = 0
for i in np.arange(0, len(np.unique(dados[:,0])), 1): ## start = 0; stop = tamanho da matriz (sem repetir os genotipos); step = 1
    count = count + 1
    matrix[i,0] = i + 1  # coluna 0 = genotipos;
    matrix[i,1] = np.mean((dados[dados[:, 0] == i + 1])[:,2])
    matrix[i,2] = np.mean((dados[dados[:, 0] == i + 1])[:,3])
    matrix[i,3] = np.mean((dados[dados[:, 0] == i + 1])[:,4])
    matrix[i,4] = np.mean((dados[dados[:, 0] == i + 1])[:,5])
    matrix[i,5] = np.mean((dados[dados[:, 0] == i + 1])[:,6])
print(matrix)

plt.figure('Médias dos genótipos para as variáveis')
plt.subplot(2,3,1) # 2x3 = 6 graficos;
plt.bar(matrix[:, 0], matrix[:,1], color="firebrick")
plt.title('Variável 1', fontsize=12)
plt.ylabel("Media")
plt.xlabel('Genótipos')

plt.subplot(2, 3, 2)
plt.bar(matrix[:, 0], matrix[:,2], color = 'firebrick')
plt.title('Variável 2', fontsize=12)
plt.ylabel("Media")
plt.xlabel('Genótipos')

plt.subplot(2, 3, 3)
plt.bar(matrix[:, 0], matrix[:,3], color = 'firebrick')
plt.title('Variável 3', fontsize=12)
plt.ylabel("Media")
plt.xlabel('Genótipos')

plt.subplot(2, 3, 4)
plt.bar(matrix[:, 0], matrix[:,4], color = 'firebrick')
plt.title('Variável 4', fontsize=12)
plt.ylabel("Media")
plt.xlabel('Genótipos')

plt.subplot(2, 3, 5)
plt.bar(matrix[:, 0], matrix[:,5], color = 'firebrick')
plt.title('Variável 5', fontsize=12)
plt.ylabel("Media")
plt.xlabel('Genótipos')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

print('2. Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável'
      ' e no eixo Y outra.') # col0 = genotipo; col1= v1; col2=v2; col3 = v3

color = ['black','blue','red','green','yellow','pink','cyan','orange','darkviolet','slategray']
loc = ["lower right"]
plt.subplot(1,3,1)
for i in np.arange(0,10,1): #start,stop,step
    plt.scatter(matrix[i,1], matrix[i,2],s=50,alpha=0.8,label = matrix[i,0],c = color[i])
plt.xlabel('Variavel 1')
plt.ylabel('Variavel 2')
plt.legend()

plt.subplot(1,3,2)
for i in np.arange(0,10,1): #start,stop,step
    plt.scatter(matrix[i,1], matrix[i,2],s=50,alpha=0.8,label = matrix[i,0], c = color[i])
plt.xlabel('Variavel 1')
plt.ylabel('Variavel 3')
plt.legend()

plt.subplot(1,3,3)
for i in np.arange(0,10,1):
    plt.scatter(matrix[i,2], matrix[i,3], s=50, alpha=0.8, label = matrix[i,0], c = color[i] )
plt.xlabel('Variavel 2')
plt.ylabel('Variavel 3')
plt.legend()
plt.show() #permite a visualização do grafico;
print('_______________________________________________________________________________________________________________')

########################################################################################################################
########################################################################################################################
