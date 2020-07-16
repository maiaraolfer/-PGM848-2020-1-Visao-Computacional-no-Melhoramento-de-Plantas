###### ------------------------------ ALUNA: MAIARA OLIVEIRA FERNANDES - 2018260157 ------------------------------ ######

print('EXERCÍCIO 03')
print('a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e '
      'a variância amostral um vetor qualquer, baseada em um loop (for).')

## BIBLIOTECA NUMPY
import numpy as np

## FUNÇÃO PARA CALCULO DE MÉDIA
def media (array):
      sum = 0
      count = 0
      for i in array:
            sum = sum + i
            count = count + 1
            mean = sum / count
      return mean

## VERIFICANDO SE A FUNÇÃO ESTÁ CORRETA
x = np.array([1,2,3,4,5])
media_x = media(x)
print('Vetor x: ' + str(x))
print('A média dos elementos do vetor x é: ' + str(media_x))

########################################################################################################################
## FUNÇÃO PARA CALCULO DE VARIÂNCIA AMOSTRAL

def variance (array):
      sum = 0
      count = 0
      sum_sq = 0
      for i in array:
            sum = sum + i
            count = count + 1
            sum_sq = sum_sq + i**2
      var = ((sum_sq-(sum**2/count))/(count-1))
      return var

## VERIFICANDO SE A FUNÇÃO ESTÁ CORRETA

print('Vetor x: ' + str(x))
var_a = variance(x)
print('A variância amostral do x é: ' + str(var_a))