import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from datetime import datetime as rtc

available_functions = ["relu","leaky_relu", "tanh", "sigmoid"]


class NeuralNetwork:
    def __init__(self, layers: List[int], epochs: int = 10, batch_size: int = 32, learning_rate: float=1e-4, validation_split: float = 0.2, verbose: int = 1, activation : str = 'relu' ):
        self._layer_structure : List[int] = layers
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.validation_split: float = validation_split
        self.verbose: int = verbose
        self._losses: Dict[str,float]={"train":[], "validation": []}
        self.activation: str = activation
        self._is_fit: bool = False
        self.__layers = None
        if self.activation not in available_functions:
            raise Exception(f"A specified activation function cannot be used. The available functions are: {available_functions}")


    def _fit(self, x:np.ndarray, y:np.ndarray) -> None:
        x, x_val, y, y_val = train_test_split(x, y, test_size = self.validation_split, random_state=42)
        self.__layers = self.__init_layers()
        logs = open(f"training_logs/{self.activation}_logs.txt", 'a')
        logs.write(f"Start: {rtc.now()}\n")
        for epoch in range(self.epochs):
            epoch_losses = []
            for i in range(1, len(self.__layers)):
                #forward pass
                x_batch = x[i:(i+self.batch_size)]
                y_batch = y[i:(i+self.batch_size)]
                pred, hidden = self._forward(x_batch)
                #calculating loss
                loss = self.calc_loss(y_batch, pred)
                epoch_losses.append(np.mean(loss**2))
                #backward pass
                self._backprop(hidden, loss)
            valid_pred, _ = self._forward(x_val)
            train_loss = np.mean(epoch_losses)
            valid_loss = np.mean(self.calc_mse(valid_pred, y_val))
            self._losses["train"].append(train_loss)
            self._losses["validation"].append(valid_loss)
        self._is_fit = True 
        logs.close()
        return


    #function used for predictions once the model has been trained
    def predict(self, X:np.ndarray) -> np.ndarray :
        if self._is_fit == False:
            raise Exception("Model is not trained yet")
        pred, hidden = self._forward(X)
        return pred
    
    def _forward(self, batch: np.ndarray) -> Tuple[ np.ndarray, List[np.ndarray]]:
        hidden = [batch.copy()]
        for i in range(len(self.__layers)):
            batch = np.matmul(batch,self.__layers[i][0]) + self.__layers[i][1]
            #the ACTIVATION FUNCTION is applied HERE    
            if i < len(self.__layers) - 1:
                batch = self.activation_function(batch)
            #hidden will later be used in backpropagation
            hidden.append(batch.copy())
        return batch, hidden

    #backpropagation function 
    def _backprop(self, hidden:List[np.ndarray], grad:np.ndarray) -> None:
        for i in range(len(self.__layers)-1, -1, -1):
            if i != len(self.__layers)-1 :
                #multiplying the output of each layer,except the final output by the DERIVATIVES of the activation function used
                grad = np.multiply(grad, self.derivative(hidden, i))
            #finding the gradient of weights and biases
            w_grad = hidden[i].T @ grad 
            b_grad = np.mean(grad)
            
            #adjusting the weights and biases
            self.__layers[i][0] -= w_grad * self.learning_rate
            self.__layers[i][1] -= b_grad * self.learning_rate

            #gradient is multiplied by the input weights to transition to next layer
            grad = grad @ self.__layers[i][0].T
        return
    
    def activation_function(self, batch: np.ndarray):
        match self.activation:
            case 'relu':
                return np.maximum(batch, 0)
            case 'leaky_relu':
                alpha = 0.1
                return np.maximum(alpha*batch, batch)
            case 'sigmoid':
                return 1/(1+np.exp(-batch))
            case 'tanh':
                return np.tanh(batch)
            #case 'softmax':
            #    return np.exp(batch)/np.sum(np.exp(batch))
            

    def derivative(self, hidden: List[np.ndarray], index: int):
        match self.activation:
            case 'relu':
                return np.heaviside(hidden[index+1],0)
            case 'leaky_relu':
                alpha = 0.01
                temp = np.ones_like(hidden[index+1])
                temp[hidden[index+1]<0] = alpha
                return temp
            case 'sigmoid':
                f = 1/(1+np.exp(-hidden[index+1]))
                return f*(1-f)
            case 'tanh':
                return 1/np.cosh(hidden[index+1])**2

    #initiation of layers. creates an array of arrays of weights and biases
    def __init_layers(self) -> List[np.ndarray]:
        layers = []
        rng = np.random.RandomState(42)
        for i in range( 1, len(self._layer_structure)):
            layers.append([
                rng.rand(self._layer_structure[i-1], self._layer_structure[i]) / 5 - .1,
                np.ones((1,self._layer_structure[i]))
            ])
        return layers
    
    #mse is used for estimating the error after 1 epoch
    def calc_mse(self, actual: np.ndarray, predicted:np.ndarray) -> np.ndarray:
        return np.mean((actual - predicted)**2)
    
    #calc_loss otputs an array of losses and is used to adjust the gradients and biases
    def calc_loss(self, actual: np.ndarray, predicted:np.ndarray) -> np.ndarray:
        return (predicted - actual)
    
    #plotting a learning curve
    def plot_learning(self) -> None :
        plt.plot(self._losses["train"], label = f"{self.activation}")
        #plt.plot(self._losses["validation"], label = f"validation {self.activation}")
        