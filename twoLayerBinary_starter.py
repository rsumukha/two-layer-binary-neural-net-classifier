'''
This file implements a two layer neural network for a binary classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    A, temp=tanh(cache["Z"])
    dZ = np.multiply(dA , (1 - np.square(A)))
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    A, temp=sigmoid(cache["Z"])
    dZ = np.multiply(dA, np.multiply(A, 1-A))
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE
    W1 = np.multiply(np.float64(0.01), np.random.randn(n_h, n_in))
    b1 = np.multiply(np.float64(0.01), np.random.randn(n_h, n_fin))
    W2 = np.multiply(np.float64(0.01), np.random.randn(n_fin, n_h))
    b2 = np.multiply(np.float64(0.01), np.random.randn(n_fin, n_fin))
    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    Z=np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function
    '''
    ### CODE HERE
    m = np.float64(Y.shape[1])
    exp = np.float64(np.sum(
        np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))))
    cost =np.float64(np.float64((-1)/m) * exp)
    return cost

def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    A=cache["A"]
    m=np.float64(A.shape[1])
    dW=np.multiply((1/m), np.dot(dZ, A.T))
    db=np.multiply((1/m), np.sum(dZ, axis=1, keepdims=True))
    dA_prev=np.dot(W.T, dZ)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    A1, cache_l1 = layer_forward(X, parameters["W1"], parameters["b1"], "tanh")
    YPred, cache_l2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
    return np.around(YPred)

def two_layer_network(train_data, train_label, val_data, val_label, net_dims, num_iterations=2000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''

    A0=train_data
    Y=train_label

    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    costs = []
    costs_val=[]

    for ii in range(num_iterations):
        # Forward propagation
        A1, cache_l1 = layer_forward(A0, parameters["W1"], parameters["b1"], "tanh")
        A2, cache_l2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
        ### CODE HERE

        # cost estimation
        cost = cost_estimate(A2, Y)
        ### CODE HERE

        # Backward Propagation
        dA2 = np.divide(A2 - Y, np.multiply(A2, 1 - A2))
        #dA2=np.multiply(np.float64(-1/np.float64(Y.shape[1])), exp)

        dA1, dW2, db2 = layer_backward(dA2, cache_l2, parameters["W2"], parameters["b2"], "sigmoid")
        dA0, dW1, db1 = layer_backward(dA1, cache_l1, parameters["W1"], parameters["b1"], "tanh")
        ### CODE HERE

        #update parameters
        parameters["W1"] = parameters["W1"] - np.multiply(learning_rate, dW1)
        parameters["b1"] = parameters["b1"] - np.multiply(learning_rate, db1)
        parameters["W2"] = parameters["W2"] - np.multiply(learning_rate, dW2)
        parameters["b2"] = parameters["b2"] - np.multiply(learning_rate, db2)

        A1, cache_l1 = layer_forward(val_data, parameters["W1"], parameters["b1"], "tanh")
        A2, cache_l2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
        ### CODE HERE
        cost_val = cost_estimate(A2, val_label)


        if ii % 10 == 0:
            costs.append(cost)
            costs_val.append(cost_val)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    
    return costs,costs_val, parameters

def stacked_autoencoder(train_data, stack_net_dims):
    parameters={}
    #layer1
    net_dims=[stack_net_dims[0], stack_net_dims[1], stack_net_dims[0]]
    H_1, parameters["W1"], parameters["b1"]=two_layer_network(train_data, train_data, net_dims, 500, 256, 0.07, stacked=True)

    #layer2
    net_dims=[stack_net_dims[1], stack_net_dims[2], stack_net_dims[1]]
    H_2, parameters["W2"], parameters["b2"]=two_layer_network(H_1, H_1, net_dims, 500, 256, 0.07, stacked=True)

    #layer3
    net_dims=[stack_net_dims[2], stack_net_dims[3], stack_net_dims[2]]
    H_3, parameters["W3"], parameters["b3"]=two_layer_network(H_2, H_2, net_dims, 500, 256, 0.07, stacked=True)

    return parameters

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    digit_range = [1,7]
    data, label, test_data, test_label = \
            mnist(noTrSamples=2400,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noTsPerClass=500)
    temp1, temp2, val_data, val_label = mnist(noTrSamples=2, noTsSamples=400,
                                             digit_range=digit_range, noTrPerClass=1,
                                             noTsPerClass=200)

    train_data = np.concatenate([data[:, :1000], data[:, 1200:2200]], axis=1)
    val_data = np.concatenate([data[:, 1000:1200], data[:, 2200:2400]], axis=1)

    train_label = np.concatenate([label[:, :1000], label[:, 1200:2200]], axis=1)
    val_label = np.concatenate([label[:, 1000:1200], label[:, 2200:2400]], axis=1)

    #convert to binary labels
    train_label[train_label==digit_range[0]] = 0
    train_label[train_label==digit_range[1]] = 1
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1
    val_label[val_label==digit_range[0]] = 0
    val_label[val_label==digit_range[1]] = 1


    n_in, m = train_data.shape
    n_fin = 1
    n_h = 500
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 0.01
    num_iterations = 500


    costs_tr, costs_val, parameters_tr = two_layer_network(train_data, train_label, val_data, val_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)


    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters_tr)
    test_Pred = classify(test_data, parameters_tr)

    trAcc = accuracy(train_Pred, train_label)
    teAcc = accuracy(test_Pred, test_label)

    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    # CODE HERE TO PLOT costs vs iterations
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("BINARY - iterations vs cost for train and validation data")
    plt.plot(costs_tr,"-g" , label='train')
    plt.plot(costs_val,":b", label='val')
    plt.legend()
    plt.show()


def accuracy(predictions, labels):
    correct_predictions = np.sum(predictions==labels)
    accuracy = 100.0 * correct_predictions / predictions.shape[1]
    return accuracy

if __name__ == "__main__":
    main()




