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

            for i in range(291):
                tmp.append(int(record[i]))

            X.append(tmp)

            tmp=[]
            for i in range(291,371):

                tmp.append(int(record[i]))

            Y.append(tmp)
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


X_train,Y_train,m_train=read_set("/Users/liuliu/Desktop/Python/flare_down/Version5/training.txt")
X_test,Y_test,m_test=read_set("/Users/liuliu/Desktop/Python/flare_down/Version5/test.txt")
print(Y_train.shape[0],Y_test.shape[0])


def feature_scaling(X_train,X_test):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    return X_train,X_test




X_train,X_test=feature_scaling(X_train.astype(float),X_test.astype(float))

def my_score(predictions,Y):



    true_pos=0
    fal_pos=0
    fal_neg=0
    m=Y.shape[0]
    for i in range(m):
        printed=False
        for j in range(len(Y[i])):

           # if(predictions[i][j]!=Y[i][j]):
              #  print(i,j,predictions[i][j],end='|')
              #  printed=True

            if(predictions[i][j]):
                if(Y[i][j]):
                    true_pos+=1
                else:
                    fal_pos+=1

            else:
                if(Y[i][j]):
                    fal_neg+=1

       # if(printed):
#           print()


    precision=true_pos/(true_pos+fal_pos)
    recall=true_pos/(true_pos+fal_neg)
    harmonic_mean=2*precision*recall/(precision+recall)
    return precision,recall,harmonic_mean



def tuning_hyper_parameters():

    hyper_learning_rate=[0.001,0.003,0.009,0.01,0.03,0.09,0.1,0.3]
    hyper_alpha=[0.9,1,2,3,4,5,6]
    hyper_n_units=range(5,46,5)

    hyper_threshold=[0.4,0.35,0.3,0.25]
    for i in hyper_alpha:
        clf = MLPClassifier(solver='adam',alpha=i, hidden_layer_sizes=(90,50,45), random_state=1,max_iter=1000,learning_rate_init=0.001)
        print("alpha =",i)
        clf.fit(X_train,Y_train)

        #print(clf.predict_proba(X_test).astype(int))
        print("train: ",my_score(clf.predict(X_train),Y_train))
        print("test: ", my_score(clf.predict(X_test), Y_test))
        #print("test: ",my_score((clf.predict_proba(X_test)).astype(int),Y_test))


tuning_hyper_parameters()



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
