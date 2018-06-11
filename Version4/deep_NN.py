import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def read_set(File):
    X=[]
    Y=[]
    with open(File, 'r') as fin:
        for line in fin:
            record=line.split(',')

            tmp=[]

            for i in range(77):
                tmp.append(int(record[i]))

            X.append(tmp)

            tmp=[]
            for i in range(77,96):

                tmp.append(int(record[i]))

            Y.append(tmp)

    X=np.array(X)
    Y=np.array(Y)

    m=Y.shape[0]

    return X,Y,m


X_train,Y_train,m_train=read_set("/Users/liuliu/My Documents/flare_down/code/Version4/training.txt")
X_dev,Y_dev,m_dev=read_set("/Users/liuliu/My Documents/flare_down/code/Version4/dev.txt")
print(Y_train.shape[0],Y_dev.shape[0])


def feature_scaling(X_train,X_dev):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_dev = scaler.transform(X_dev)
    return X_train,X_dev




X_train,X_dev=feature_scaling(X_train.astype(float),X_dev.astype(float))

def my_score(predictions,Y):

    true_pos=0
    fal_pos=0
    fal_neg=0
    m=Y.shape[0]
    for i in range(m):

        for j in range(len(Y[i])):


            if(predictions[i][j]):
                if(Y[i][j]):
                    true_pos+=1
                else:
                    fal_pos+=1

            else:
                if(Y[i][j]):
                    fal_neg+=1


    return true_pos/(true_pos+fal_pos),true_pos/(true_pos+fal_neg)



def tuning_hyper_parameters():
    r=-4*np.random.rand(6)
    r.sort()
    hyper_learning_rate=[0.001,0.003,0.009,0.01,0.03,0.09,0.1,0.3]
    hyper_alpha=[1,1.2,1.4,1.6,1.8,2,2.2]
    hyper_n_units=range(5,46,5)

    hyper_reg=[3,3.5,4,4.5,5]
    hyper_threshold=[0.4,0.35,0.3,0.25]
    #for i in hyper_threshold:
    clf = MLPClassifier(solver='adam',alpha=1.1, hidden_layer_sizes=(90,50,45), random_state=1,max_iter=1000,learning_rate_init=0.001)
    #print("alpha =",i)
    clf.fit(X_train,Y_train)


    print("train: ",my_score(clf.predict(X_train),Y_train))
    print("dev: ", my_score(clf.predict(X_dev), Y_dev))



tuning_hyper_parameters()



def plot_tuning():   #plot the learning curve to tune the network
    x_alpha=[]
    y_n_units=[]
    z_score_train=[]
    z_score_dev=[]
    hyper_learning_rate=[0.001,0.003,0.009,0.01,0.03,0.09,0.1,0.3]
    hyper_alpha=[1,1.2,1.4,1.6,1.8,2,2.2]
    hyper_n_units=range(5,46,5)

    hyper_reg=[3,3.5,4,4.5,5]
    hyper_threshold=[0.4,0.35,0.3,0.25]
    for i in hyper_alpha:
        for j in hyper_n_units:
            clf = MLPClassifier(solver='adam', alpha=0.6, hidden_layer_sizes=(j), random_state=0, max_iter=5000, learning_rate_init=i)
            print("learning_rate=", i,"hidden units=",j)
            clf.fit(X_train,Y_train)

            x_alpha.append(i)
            y_n_units.append(j)
            print(clf.score(X_train, Y_train)*100,'%')
            print(clf.score(X_dev,Y_dev)*100,'%')
            z_score_train.append(clf.score(X_train, Y_train)*100)
            z_score_dev.append(clf.score(X_dev,Y_dev)*100)


    fig = plt.figure()
    ax = fig.gca(projection='3d')


    ax.plot(x_alpha, y_n_units, z_score_train, label='train')
    ax.plot(x_alpha, y_n_units, z_score_dev,label="dev")
    ax.legend()

    plt.show()
