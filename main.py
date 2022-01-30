import numpy as np
from model import *
from scipy.optimize import minimize
import pandas as pd
 
 
df_train = pd.read_csv('Dataset/train.csv')
x= df_train.drop(['label'],axis=1) #mX784
y = df_train['label'] #mX1
m =  len(x.axes[0])
nodes_input = len(x.axes[1])  #784 without bias
x = x.to_numpy()
#standardising inputs
x=x/255
y= y.to_numpy()
y_new = np.zeros((m,10))
for i in range(m):
    y_new[i,y[i]]=1
#ideally the no of nodes in the hidden layer = 2/3 of the nodes in input and output
nodes_hidden = 532
nodes_output = 10
#532X785
initial_theta1 = rand_theta(nodes_hidden, nodes_input+1) #nodes_input + 1 to accommodate theta values for bais
#10X533
intitial_theta2 = rand_theta(nodes_output, nodes_hidden+1) #nodes_hidden + 1 to accommodate bias for the hidden layer
 
# Unrolling parameters into a single column vector
theta_vec = np.concatenate((initial_theta1.flatten(), intitial_theta2.flatten())) # 532*785 + 533*10
iterations = 100
lambda_reg = 0.1
nn_args = (nodes_input, nodes_hidden, nodes_output, x, y_new, lambda_reg)
 
# Calling minimize function to minimize cost function and to train weights
weights = minimize(neural_network, x0=theta_vec, args=nn_args,options={'disp': True, 'maxiter': iterations}, method="L-BFGS-B", jac=True)
 
weights_trained = weights["x"]  # Trained Theta is extracted
theta1 = np.reshape(weights_trained[:nodes_hidden * (nodes_input + 1)], (nodes_hidden, nodes_input + 1))  #532X785
theta2 = np.reshape(weights_trained[nodes_hidden * (nodes_input + 1):],(nodes_output, nodes_hidden + 1))  #10X533
pred = predict(theta1, theta2, x)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y) * 100)))
 
# Evaluating precision of our model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y[i]:
        true_positive += 1
false_positive = len(y) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))
 
# Saving Thetas in .txt file
np.savetxt('theta1.txt', theta1, delimiter=' ')
np.savetxt('theta2.txt', theta2, delimiter=' ')