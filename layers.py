import numpy as np
np.random.seed(1)


class Conv2D(object):
    """Class for creating a Conv2D layer.
    INPUT:
    filter_shape: Shape of kernel as a tuple (f, f)
    num_filters: Number of feature maps in output
    stride: stride length, defaults to 1
    padding: padding, defaults to 0
    
    OUTPUT:
    """
    
    
    def __init__(self, filter_shape, num_filters, stride = 1, padding = 0, activation = 'relu'):
        self.f_h = filter_shape[0]
        self.f_w = filter_shape[1]
        self.f_d = filter_shape[2]

        
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.W = np.random.normal(0, 1.0/(self.f_h * self.f_w *self.num_filters),(self.f_h, self.f_w, self.f_d, self.num_filters) )
        self.b = np.random.normal(0, 1.0/self.num_filters, (1,1,self.f_d, self.num_filters))
        self.activation = 'relu'
        self.type = "Conv2D"
        
        
    def zero_pad(self, X):
        """Pads with zeros the numpy array as input
        INPUT:
        X : numpy array (Image) of size (num_samples, h_in, w_in, d_in)
        
        
        OUTPUT:
        X_pad : padded numpy array"""
        
        X_pad = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = 0)
        
        return X_pad
    
    def conv_single_slice(self, slice_prev, depth):
        """Single step of convolution using slice_prev which generates one pixel as output
        INPUT:
        slice_prev: Input slice for one kernel, shape = (f, f, d_in)
        depth: Index of the output depth
      
        OUTPUT:
        conv_result: shape=(1,1,1)"""
        
        conv_result = np.sum(np.multiply(slice_prev, self.W[:,:,:, depth])) + self.b[:,:,:, depth]
        
        return conv_result
    
    def relu(self, x):
        return x * (x>0)
    
    def relu_prime(self, x):
        return x>0
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        return np.multiply(sigmoid(x), 1-sigmoid(x))
    
    def forward(self, X):
        """Single forward pass for input image
        INPUT:
        X: Input sets of images, shape = (num_samples, h_in, w_in, d_in)
        
        OUTPUT:
        Y: Output feature maps, shape = (num_samples, h_out, w_out, d_out)
        X: as cache to be used in backpropagation
        """
        
        (no_samples, h_in, w_in, d_in) = X.shape
        
        (f, f, d_in, d_out) = self.W.shape
        
        h_out = int((h_in - f + 2*self.padding)/self.stride) + 1
        w_out = int((w_in - f + 2*self.padding)/self.stride) + 1
        
        Y = np.zeros((num_samples, h_out, w_out, d_out))
        X_pad = self.zero_pad(X)
        for i in range(num_samples):
            
            for j in range(h_out):
                
                for k in range(w_out):
                    
                    for c in range(d_out):
                        left = self.stride * k
                        right = left + f
                    
                        up = self.stride * j
                        bottom = up + f
                    
                    
                        Y[i, j, k, c] = self.conv_single_slice(X_pad[i, up:bottom, left:right, :], c)
                        
        if(self.activation == "relu"):
            Y = self.relu(Y)
            
        elif(self.activation == "sigmoid"):
            Y = self.sigmoid(Y)
            
    
                        
        cache = (X, Y)
        return Y, cache
    
    def backward(self, dZ, cache):
        """Function for calculating and hence even updating the gradients and values of weights and biases
        INPUT: 
        dZ: This is the gradient of cost function with respect to the activation in output layer, 
            shape= (num_samples, h_out, w_out, d_out)
        cache: cache from the forward function to be used in backprop
        
        OUTPUT:
        dW = Gradient of loss function with respect to weights, shape =(f, f, d_in, d_out)
        db = Gradient of loss function with respect to biases, shape = (1, 1, 1, d_out)
        dA_priv = dZ value to be passed to the previous layer during backprop
        """
        
        X_prev, Y = cache
        (num_samples, h_in, w_in, d_in) = X_prev.shape
        
        dA_prev = np.zeros(X_prev.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        
        (num_samples, h_out, w_out, d_out) = dZ.shape
        
        X_prev_pad = self.zero_pad(X_prev)
        dA_prev_pad = self.zero_pad(dA_prev)
        
        if(self.activation == "relu"):
            dZ = np.multiply(dZ, self.relu_prime(dZ))
        elif(self.activation == "sigmoid"):
            dZ = np.multiply(dZ, self.sigmoid_prime(dZ))
        
        
        for i in range(num_samples):
            x_prev_pad = X_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for j in range(h_out):
                
                for k in range(w_out):
                    
                    for c in range(d_out):
                        
                        up = self.stride * j 
                        down = self.stride * j + self.f_h
                        
                        left = self.stride*k
                        right = self.stride*k + self.f_w
                        
                        
                        da_prev_pad[up:down, left:right, :] += self.W[:,:,:,c]* dZ[i, j, k, c]
                        
                        dW[:, :, :, c] += x_prev_pad[up:down, left:right, :] * dZ[i, j, k, c]
                        
                        db += dZ[i, j, k, c]
                        
                    
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            
            dW = dW / num_samples
            db = db / num_samples
        
        
        
        return dW,db, dA_prev



class Dense(object):
    
    def __init__(self, num_features, num_samples, output_shape, activation = "relu" ):
        self.num_features = num_features
        self.num_samples = num_samples
        self.output_shape = output_shape
        self.activation = activation
        self.W = float(1.0/(num_features*output_shape))*np.random.randn(num_features, output_shape)
        self.b = float(1.0/output_shape)*np.random.randn(1, output_shape)
        self.type = "Dense"
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        return np.multiply(self.sigmoid(x), 1-self.sigmoid(x))
    
    def relu(self, x):
        return x * (x>0)
    
    def relu_prime(self, x):
        return x>0
        
    def forward(self, X):
        
        
        Y = np.dot(X, self.W) + self.b
        cache = (Y, X)
        
        if(self.activation == "relu"):
            Y = self.relu(Y)
        elif(self.activation == "sigmoid"):
            Y = self.sigmoid(Y)
        
        return Y, cache
    
    def backward(self, dZ, cache):
        
        (num_samples, f_out) = dZ.shape
        
        dA = np.zeros((self.num_samples, self.num_features))
        dW = np.zeros((self.num_features, f_out))
        (Y, X) = cache
        db = np.zeros((1, f_out))
        
        for i in range(num_samples):
            
            
            if(self.activation == "relu"):
                dZ = dZ*self.relu_prime(Y)
            elif(self.activation == "sigmoid"):
                dZ = dZ*self.sigmoid_prime(Y)
            
            
            dA[i,:] = np.dot(dZ[i, :], self.W.T)
            dW += np.outer(X[i, :], dZ[i, :])
            db += dZ[i, :]
            
        dW = dW / self.num_samples
        db = db / self.num_samples
        
        return dW, db, dA

    
    

class flatten(object):
    """Flattens the input element"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.type = "flatten"

        pass
    
    def forward(self, X):
        
        (num_samples, h, w, d) = X.shape
        return X.reshape(num_samples, h*w*d), 
    
    def backward(self, dZ):
        return dZ.reshape(self.input_shape)
    



class MaxPool2D(object):
    """Max Pool layer 2D with the following input parameters
    INPUT"""
    
    def __init__(self, kernel_shape = (2,2), stride = (1,1)):
        self.k_h = kernel_shape[0]
        self.k_w = kernel_shape[1]
        
        self.stride = stride
        self.type = "MaxPool2D"
        
    def mask_from_slice(self, x):
        """Creates mask from the slice putting ones at the place the value of which goes to the next"""
        
        return x==np.max(x)
    
    
    
    def forward(self, X):
        """Calculates the output feature maps for a given input layer
        INPUT:
        """
        
        (num_samples, h_in, w_in, d_in) = X.shape
        h_out = int((h_in - self.kernel_shape[0] )/self.stride) + 1
        w_out = int((w_in - self.kernel_shape[1] )/self.stride) + 1
        
        d_out = d_in
        
        Y = np.zeros((num_samples, h_out, w_out, d_out))
        cache = np.zeros(X.shape)
        
        for i in range(num_samples):
            
            for j in range(h_out):
                
                for k in range(w_out):
                    
                    for d in range(d_out):
                        
                        up = j * self.stride 
                        down = up + self.kernel_shape[0]
                        
                        left = k * self.stride 
                        right = left + self.kernel_shape[1]
                        
                        
                        Y[i, j, k, d] = np.max(X[i, up:down, left:right, d])
                        
        cache = X               
        return Y, cache
    
    
    def backward(self, dZ, cache):
        
        """Implements backpropagation through the layer
        INPUT:
        dZ: Gradient of loss function with respect to the activations of output feature maps
        cache: Cache from the forward function to help us in backpropagation"""
        
        
        
        X_prev = cache
        
        (num_samples, h_in, w_in, d_in) = X_prev.shape
        (num_samples, h_out, w_in, d_in) = dZ.shape
        
        dA = np.zeros(X_prev.shape)
        
        for i in range(num_samples):
            
            for j in range(h_out):
                
                for k in range(w_out):
                    
                    for c in range(d_out):
                        
                        up = j * self.stride
                        down = up + self.k_h
                        
                        left = k * self.stride
                        right = left + self.k_w
                        
                        mask = self.mask_from_slice(X_prev[i, up:down, left:right, c])
                        
                        dA[i, up:down, left:right, c] += np.multiply(dZ[i, j, k, c], mask)
        return dA
       
                        
                        
                        
        
       
        
        
        
        
                        
        
                    
                    
                    
                    
        
        
        
        
    
    
    
                
            
                
                
            
            
            
            
            
        
        
        
        
    
