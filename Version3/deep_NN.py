import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.preprocessing import StandardScaler

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

    # newY=[]
    # for i in Y:
    #     tmp=[0]*410
    #
    #     tmp[i]=1
    #     newY.append(tmp)
    #

    #
    #
    # Y=np.array(newY)

    X=np.array(X)
    Y=np.array(Y)

    m=Y.shape[0]

    return X,Y,m


X_train,Y_train,m_train=read_set("/Users/liuliu/My Documents/flare_down/code/Version3/training.txt")
X_test,Y_test,m_test=read_set("/Users/liuliu/My Documents/flare_down/code/Version3/dev.txt")



def feature_scaling(X_train,X_test):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    return X_train,X_test





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


def my_score(predictions,Y):
    newY=[]
    for i in Y:
        tmp=[0]*19
        tmp[i]=1
        newY.append(tmp)

    Y=np.array(newY)

    newY = []
    for i in predictions:
        tmp=[0]*19
        tmp[i]=1
        newY.append(tmp)

    predictions=np.array(newY)

    accuracy=0
    m=Y.shape[0]
    for i in range(m):
        for j in range(len(Y[i])):
            if(predictions[i][j]==Y[i][j]):
                accuracy+=1


    return accuracy/(m*19)*100



def tuning_hyper_parameters():
    r=-4*np.random.rand(6)
    r.sort()
    hyper_alpha=[0.00099,0.001,0.0012,0.0014,0.0016,0.0018]
    hyper_n_units=range(5,46,5)

    hyper_reg=[3,3.5,4,4.5,5]



clf = MLPClassifier(solver='adam', alpha=1.1, hidden_layer_sizes=(90,50,45), random_state=1,max_iter=1000,learning_rate_init=0.001)

clf.fit(X_train,Y_train)

print("performance on training set",my_score(clf.predict(X_train),Y_train),'%')
print("performance on dev set",my_score(clf.predict(X_test),Y_test),'%')





def plot_tuning():
    x_alpha=[]
    y_n_units=[]
    z_score_train=[]
    z_score_test=[]
    for i in hyper_alpha:
        for j in hyper_n_units:
            clf = MLPClassifier(solver='adam', alpha=0.6, hidden_layer_sizes=(j), random_state=0, max_iter=5000, learning_rate_init=i)
            print("learning_rate=", i,"hidden units=",j)
            clf.fit(X_train,Y_train)

            x_alpha.append(i)
            y_n_units.append(j)
            print(clf.score(X_train, Y_train)*100,'%')
            print(clf.score(X_test,Y_test)*100,'%')
            z_score_train.append(clf.score(X_train, Y_train)*100)
            z_score_test.append(clf.score(X_test,Y_test)*100)


    fig = plt.figure()
    ax = fig.gca(projection='3d')


    ax.plot(x_alpha, y_n_units, z_score_train, label='train')
    ax.plot(x_alpha, y_n_units, z_score_test,label="test")
    ax.legend()

    plt.show()
