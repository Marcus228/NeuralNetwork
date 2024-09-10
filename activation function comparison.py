import matplotlib.pyplot as plt
import numpy as np

def plot_func(x,y,title):
    plt.plot(x,y)
    plt.grid(True)
    plt.title(title)
    plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def leaky_ReLU(x,alpha = 0.1):
    return np.maximum(alpha*x,x)

def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

x = np.linspace(-10,10,100)
plot_func(x,leaky_ReLU(x),'leaky ReLU')
plot_func(x,ReLU(x),'ReLU')
plot_func(x,sigmoid(x),'sigmoid')
plot_func(x,tanh(x),'tanh')
plot_func(x,softmax(x),'softmax')