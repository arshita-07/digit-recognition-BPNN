import numpy as np
 
 
def neural_network(nn_params, nodes_input, nodes_hidden, nodes_output, x, y, lamb):
    theta1 = np.reshape(nn_params[:nodes_hidden * (nodes_input + 1)],(nodes_hidden, nodes_input + 1)) #532x785
    theta2 = np.reshape(nn_params[nodes_hidden * (nodes_input + 1):],(nodes_output, nodes_hidden + 1)) #10x533
    [m,n] = x.shape
    mat_ones = np.ones((m, 1))
    x = np.append(mat_ones, x, axis=1)  #bias
    #forward propagation to calc cost
    a1 = x #has a col of ones for bias added already mx785
    z2 = np.dot(x, theta1.transpose()) #mx532
    a2 = activation(z2)  ##mx532
    a2 = np.append(mat_ones, a2, axis=1)  #mx533 adding bias node
    z3 = np.dot(a2, theta2.transpose()) #mx10
    a3 = activation(z3) #mx10
    j = (1 / m) * (np.sum(np.sum(-y * np.log(a3) - (1 - y) * np.log(1 - a3)))) + (lamb / (2 * m)) * (sum(sum(pow(theta1[:, 1:], 2))) + sum(sum(pow(theta2[:, 1:], 2))))
    #backpropagation to calc gradient
    delta3 = a3 - y
    delta2 = np.dot(delta3, theta2) * a2 * (1 - a2)
    delta2 = delta2[:, 1:]
    theta1[:, 0]=0 #this is done because when j=0 we dont add lamb/m*theta1
    del2=(1/m)*np.dot(delta2.transpose(),a1)+(lamb/m)*theta1 #532x785
    theta2[:, 0]=0
    del3=(1 / m)*np.dot(delta3.transpose(),a2) + (lamb/m)*theta2 #10x533
    grad = np.concatenate((del2.flatten(), del3.flatten()))
 
    return j, grad

def activation(z):
    return 1 / (1 + np.exp(-z))

 
def rand_theta(rows, cols):
    epsilon = 0.15
    c = np.random.rand(rows, cols) * (2 * epsilon) - epsilon 
    return c

def predict(theta1, theta2, x):
    m = x.shape[0]
    one_matrix = np.ones((m, 1))
    x = np.append(one_matrix, x, axis=1)
    z2 = np.dot(x, theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2))
    a2 = np.append(one_matrix, a2, axis=1)
    z3 = np.dot(a2, theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3))
    p = (np.argmax(a3, axis=1))
    return p