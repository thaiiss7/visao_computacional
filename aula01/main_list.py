import math as m
import random as rd

def S(x, w ,b):  #neurônio
    result = 0
    for i in range(len(x)):
        result += (x[i] * w[i])
    result = result + b
    return result
        

def sigmoid(S):  #função de ativação
    return 1/(1 + m.e ** (-S))

# [0, 0, 1]

t = 1.00

#variáveis fixas:
lr = 0.05
# x = input
x = [2, 4.1, 0.3 , 1.6, 2.3]

#mudam:
# w = pesos
w = [0.24550877909173696, 0.8457522326392307, 0.5045879196405884, 0.5514694883204624, 0.13407673554031596]
# b = viés
b = rd.random()

# em cada época ele treina com todos os dados
epochs = 100
for i in range(epochs):
    t = 1.0
    
    soma = S(x, w ,b)
    y = sigmoid(soma)
    
    error = 0.5 * pow((t - y), 2)
    
    for j in range(len(x)):
        derivada_vies = (y - t) * y*(1 - y)* x[j]
        derivada_peso = (y - t) * y*(1 - y)
    
    w[j] = w[j] - lr * derivada_peso
    b = b - lr * derivada_vies
    
    print(f"Época: {i}, Saída: {sigmoid(S(x,w,b))}, Erro: {error}")
    pass
print(w)