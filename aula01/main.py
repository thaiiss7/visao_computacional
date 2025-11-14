import math as m
import random as rd

def S(x, w ,b):  #neurônio
    return (x * w) + b

def sigmoid(S):  #função de ativação
    return 1/(1 + m.e ** (-S))

# [0, 0, 1]

t = 1.00

#variáveis fixas:
lr = 0.05
# x = input
x = 1.6

#mudam:
# w = pesos
w = rd.random()
# b = viés
b = rd.random()

epochs = 100
for i in range(epochs):
    t = 1.0
    soma = S(x, w ,b)
    y = sigmoid(soma)
    
    error = 0.5 * pow((t - y), 2)
    derivada_vies = (y - t) * y*(1 - y)* x
    derivada_peso = (y - t) * y*(1 - y)
    
    w = w - lr * derivada_peso
    b = b - lr * derivada_vies
    
    print(sigmoid(S(x,w,b)), error)
    pass
