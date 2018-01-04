import numpy as np
import random

class Model():
    
    def __init__(self, train_data, val_data, layers, batch_size = 100, learning_rate = 0.0001, error = "MSE"):
        
        self.X_train = train_data[0]
        self.Y_train = train_data[1]
        self.X_val = val_data[0]
        self.Y_val = val_data[1]
        self.num_layers = len(layers)
        self.layers = layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.error = error
    
    
    def predict(self, X):
        
        output = X
        for layer in self.layers:
            
            output, temp_cache = layer.forward(output)
            
        output = output>0.5           
        return output
    
    
    def fit(self, epochs = 1):
        
        for epoch in range(epochs):
            
            
            
            
            cache = []
            output = self.X_train
            for layer in self.layers:
                output, temp_cache = layer.forward(output)
                cache.append(temp_cache)
            
            Y_calc = output
            
                
        
        
            if(self.error == "MSE"):
                dZ = self.Y_train - Y_calc
                
                for i in range(self.num_layers):
                    layer = self.layers[self.num_layers - i - 1]
                    
                    if((layer.type == "Conv2D") or (layer.type == "Dense")):
 
                       print("We do come here")

                       dW, db, dZ = layer.backward(dZ, cache[self.num_layers - i - 1])
                       
                       layer.W -= self.learning_rate * dW
                       layer.b -= self.learning_rate * db
                        
                    if(layer.type == "MaxPool2D"):
                        dZ = layer.backward(dZ, cache[self.num_layers - i - 1])
                    
                
                
                print("Epoch {} completed. Accuracy = {}".format(epoch, self.accuracy()))


        
                
    def accuracy(self):
        Y_calc = self.predict(self.X_train)
        accuracy = 0
        
        for i in range(len(self.X_train)):
            if((Y_calc[i] == self.Y_train[i]).all()):
                accuracy +=1;
            
            
        return float(accuracy)/len(self.X_train)
            
            
                    
                    
                    
