import numpy as np
import time

def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialise_parameters(n_x,n_h,n_y):  #initialise W1,b1,W2,b2
    np.random.seed(int(time.time()))
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))



    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters


def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    tmp=np.exp(Z2)
    A2=tmp/np.sum(tmp,axis=0,keepdims=True)


    cache={"Z1":Z1,
           "A1":A1,
           "Z2":Z2,
           "A2":A2
           }
    return A2,cache

def compute_cost(A2,Y,parameters):

    m=Y.shape[1]
    logprobs=np.multiply(Y,np.log(A2))+np.multiply(1-Y,np.log(1-A2))
    cost=-1/m*np.sum(logprobs)
    cost=np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost


def softmax_cost(A2,Y,parameters):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2))
    cost = -1 / m * np.sum(logprobs)
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1=cache["A1"]
    A2=cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.transpose())
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.transpose(), dZ2), 1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.transpose())
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads={"dW1":dW1,
           "dW2":dW2,
           "db1":db1,
           "db2":db2
           }
    return grads

def update_parameters(parameters,grads,learning_rate):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1=grads["db1"]
    db2=grads["db2"]

    W1-=dW1*learning_rate
    W2-=dW2*learning_rate
    b1-=db1*learning_rate
    b2-=db2*learning_rate

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


    n_x = X.shape[0]
    n_y = Y.shape[0]


    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialise_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = softmax_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=0.4)

        ### END CODE HERE ###

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters




def read_set(File):
    X=[]
    Y=[]
    with open(File, 'r') as fin:
        for line in fin:
            record=line.split(',')
            tmp=[]

            for i in range(len(record)-2):
                tmp.append(int(record[i]))

            X.append(tmp)
            Y.append(int(record[-2]))

    newY=[]
    for i in Y:
        tmp=[0]*410

        tmp[i]=1
        newY.append(tmp)

    X=np.array(X)
    X=X.transpose()

    Y=np.array(newY)
    Y=Y.transpose()
    m=Y.shape[1]
    return X,Y,m


X_train,Y_train,m_train=read_set("/Users/liuliu/Desktop/Python/flare_down/Version2/training.txt")
X_test,Y_test,m_test=read_set("/Users/liuliu/Desktop/Python/flare_down/Version2/test.txt")

parameters=nn_model(X_train, Y_train, 10, num_iterations=10000, print_cost=True)

# print("W1", parameters["W1"])
# print("W2",parameters["W2"])
# print("b1",parameters["b1"])
# print("b2",parameters["b2"])
# print(A2)
# print(Y)
# print(compute_cost(A2,Y,parameters))
# grads=backward_propagation(parameters,cache,X,Y)
# print(grads["db1"])

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)

    A2, cache = forward_propagation(X, parameters)

    A2=A2.transpose()
    predictions=[]
    for i in range(A2.shape[0]):
        rA2=np.array(A2[i])
        tmp=[0]*rA2.size
        for j in range(rA2.size):
            if(rA2[j]==rA2.max()):
                tmp[j]=1
                break

        predictions.append(tmp)

    predictions=np.array(predictions)
    predictions=predictions.transpose()


    return predictions


def calculate_accuracy(predictions,Y):

    accuracy=0
    m=Y.shape[1]
    Y=Y.transpose()
    predictions=predictions.transpose()
    for i in range(predictions.shape[0]):
        accuracy+=((predictions[i]==Y[i]).all())

    return accuracy/m*100

predictions=predict(parameters,X_train)
print("performance on training set:",calculate_accuracy(predictions,Y_train),'%')
predictions=predict(parameters,X_test)
print("performance on dev set",calculate_accuracy(predictions,Y_test),'%')