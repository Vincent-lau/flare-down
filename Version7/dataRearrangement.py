from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import codecs
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def csv_to_text(): #need to deal with "Incontinence urine"
    with codecs.open("/Users/liuliu/myDocuments/flare_down/code/fd-export2.csv", 'r', encoding="utf-8") as f1:
        with codecs.open("/Users/liuliu/myDocuments/flare_down/code/Version6/txtFileOfCsv.txt", "w",encoding="utf-8") as f2:

            preline=""
            for line in f1:

                if(not "  urine" in line and not "  central" in line and not "  obstructive" in line and not "  dry" in line):
                    f2.write(preline)
                    preline = line.rstrip('\n')
                    if (preline[-1] != ','):
                        preline += ','
                    preline += '\n'
                else:
                    preline=preline.rstrip(",\n")
                    line=line.lstrip(' ')
                    preline+=(' '+line.rstrip('\n')+",\n")

                    f2.write(preline)
                    preline=""

#
# csv_to_text()
def remove_quotes():  #remove quotes around "Incontinence urine"
    with codecs.open("/Users/liuliu/myDocuments/flare_down/code/Version6/txtFileOfCsv.txt", 'r', encoding="utf-8") as fin:
        with codecs.open("/Users/liuliu/myDocuments/flare_down/code/Version6/remove_quotes.txt", "w",encoding="utf-8") as fout:
            for line in fin:
                record=line.split(',')


                for i in record:
                    if('"' in i):
                        i=i.replace('"','').replace('“','')
                    fout.write(i)
                    if(i!='\n'):
                        fout.write(',')


# ----------------------------------------------------------------------------
def delete_trackableID():    #delete the column trackableID
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/txtFileOfCsv.txt", 'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/delete_trackableID.txt", 'w',encoding="utf-8") as fout:
            for line in fin:
                record=line.split(',')
                for i in range(len(record)):
                    if (i != 5):

                        fout.write(record[i])
                        if (record[i] != '\n'):
                            fout.write(',')


#----------------------------------------------------------------------------
def rearrange():           #rearrange data such that all information under the same user will be grouped
    with open ("/Users/liuliu/myDocuments/flare_down/code/Version6/delete_trackableID.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/rearrange.txt",'w',encoding="utf-8") as fout:
            preID = ""
            for line in fin:

                record=line.split(',')


                if(record[0]!=preID):
                    preID=record[0]
                    fout.write('\n')

                    for i in range(4):
                        if(len(record)<4):
                            print(record)
                        fout.write(record[i])

                        if(record[i]!='\n'):
                            fout.write(',')
                    fout.write('\n')

                for i in range(4,len(record)):
                    fout.write(record[i])
                    if (record[i] != '\n'):
                        fout.write(',')


#----------------------------------------------------------------------------
def sort_by_date():             #sort every group according to dates and delete all date afterwards
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/rearrange.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/sort_by_date.txt",'w',encoding="utf-8") as fout:
            module=[]
            for line in fin:
                record=line.split(',')
                if("==" in record[0]):
                    tmp=module[1:-1]

                    tmp.sort(key=lambda tmp: tmp[0])

                    module[1:-1]=tmp

                    for row in module:
                        for i in range(len(row)):
                            flag=False
                            if(not "20" in row[i] or i!=0):
                                flag=True
                                fout.write(row[i])
                            if(row[i]!='\n' and flag):
                                fout.write(',')

                    module.clear()
                    module.append(record)
                else:
                    module.append(record)


# ----------------------------------------------------------------------------
def discard_noSymptoms():   #discard groups with no symptoms i.e. no y value
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/sort_by_date.txt", 'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/discard_noSymptoms.txt", 'w',encoding="utf-8") as fout:
            module = []
            for line in fin:
                flag = False
                if ("==" in line):
                    for i in module:
                        record=i.split(',')
                        if(record[0]=="Symptom" and int(record[-2])!=0 ):
                            flag=True
                            break

                    if (flag):
                        for i in module:
                            fout.write(i)
                        fout.write('\n')

                    module = []
                    module.append(line)
                else:
                    module.append(line)


# ----------------------------------------------------------------------------
def discard_treatment_weather_zeroSymptom():  #discard all treamtment, weather, and symptoms with zeros
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/discard_noSymptoms.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/discard_treatment_weather_zeroSymptom.txt",'w',encoding="utf-8") as fout:
            for line in fin:
                record=line.split(',')
                flag=True

                if(record[0]=='Weather' or record[0]=="Treatment"):
                    flag=False

                if(record[0]=="Symptom" and int(record[-2])==0):
                    flag=False

                if(flag):
                    fout.write(line)


#----------------------------------------------------------------------------
def seperate_groups():  #seperate groups such that every group ends with some symptoms and discard those without a symptom at the end
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/discard_treatment_weather_zeroSymptom.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/seperate_groups.txt",'w',encoding="utf-8") as fout:
            module = []
            symptom_flag=False
            for line in fin:
                if ("==" in line):
                    till_last_symptom = []
                    for row in module:

                        record= row.split(',')
                        if (record[0] == "Symptom"):
                            symptom_flag=True
                            till_last_symptom.append(row)


                        elif(not "==" in row):
                            if(symptom_flag):
                                fout.write(module[0])
                                for i in till_last_symptom:
                                    fout.write(i)

                                fout.write('\n')
                                till_last_symptom.clear()

                                symptom_flag=False

                            till_last_symptom.append(row)

                    module.clear()
                    till_last_symptom.clear()
                    module.append(line)

                elif (line != ''):
                    module.append(line)


# ----------------------------------------------------------------------------
def read_word_vecs(word_vecs):

    with codecs.open(word_vecs, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        first = True
        for line in f:
            if (first):
                first = False
                continue

            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map




def generate_list():  # generate lists that has all the conditions, tags and symptoms in them for the conversion to averaged word vectors
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/sort_by_date.txt", 'r',encoding="utf-8") as fin:

        countries = {"": 0}
        tags_list = []
        symptoms_list = []
        conditions_list = []
        counter_conunrties = 0
        for line in fin:

            record = line.split(',')
            if ("==" in line):
                if (not record[3] in countries):
                    counter_conunrties += 1
                    countries[record[3]] = counter_conunrties

            elif (record[0] == "Condition"):

                if (not record[1] in conditions_list):
                    conditions_list.append(record[1])

            elif (record[0] == "Tag"):

                if (not record[1] in tags_list):
                    tags_list.append(record[1])

            elif (record[0] == "Symptom"):
                if (not record[1] in symptoms_list):
                    symptoms_list.append(record[1])
        return conditions_list, tags_list, symptoms_list,countries



def sentence_to_avg(word_to_vec_map,X):  # convert a particular string of condition to a word vector by averaging every word in the string
    X_vec_dic = {}
    X_vec_list = []

    for sentence in X:
        words = sentence.lower().split(' ')
        avg = 0
        for w in words:
            if (not w in word_to_vec_map):

                break
            else:
                avg += word_to_vec_map[w]

        else:
            avg /= len(words)
            X_vec_dic[sentence] = avg
            X_vec_list.append(avg)

    return X_vec_dic, X_vec_list



# ----------------------------------------------------------------------------

def visualisation(x1):  # applying PCA to visualise the data

    pca = PCA(n_components=2)
    newX1 = pca.fit_transform(x1)
    print("PCA variance sum of word vectors", np.sum(pca.explained_variance_ratio_))

    plt.scatter(newX1[:, 0], newX1[:, 1], c="red")

    plt.show()




def plot_Kmeans_error(X):  # plot k-means error to choose number of centroids
    error = []
    for i in range(80, 200):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        print(kmeans.inertia_, i)
        error.append([i, kmeans.inertia_])
        # if(i==16 or i==14):
        #    visualisation(X,kmeans.cluster_centers_)

    error = np.array(error)
    plt.scatter(error[:, 0], error[:, 1])
    plt.show()






# ----------------------------------------------------------------------------

def get_clusters(X_list, X_dic,n_centroids):  # print every entry in the list and its corresponding centroid
    X_list = np.array(X_list)
    kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(X_list)
    print("kmeans.inertia_",kmeans.inertia_)
    clusters = []

    for i in range(len(X_dic.keys())):
        clusters.append([list(X_dic.keys())[i], np.squeeze(kmeans.predict(X_list[i].reshape(1, -1)))])

    clusters.sort(key=lambda clusters: clusters[1])
    return clusters




def write_dic(countries,conditions,tags,symptoms):
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/dic.txt",'w',encoding="utf-8") as fout:
        fout.write("COUNTRY\n")
        for s in countries:
            fout.write(s+','+str(countries[s])+",\n")

        fout.write("CONDITION\n")
        for c in conditions:
            fout.write(str(np.squeeze(c[0]))+','+str(np.squeeze(c[1]))+",\n")
        fout.write("TAG\n")
        for c in tags:
            fout.write(str(np.squeeze(c[0])) + ',' + str(np.squeeze(c[1])) + ",\n")

        fout.write("SYMPTOM\n")
        for c in symptoms:
            fout.write(str(np.squeeze(c[0])) + ',' + str(np.squeeze(c[1])) + ",\n")


def generate_dic(words, word_to_vec_map,n_condition_clusters,n_tag_clusters,n_symptom_clusters):   #generate by clustering all symptoms, conditions  and tags

    print("-------------------word vectors loaded-------------------")
    conditions_list, tags_list, symptoms_list, countries = generate_list()

    conditions_vec_dic, conditions_vec_list = sentence_to_avg(word_to_vec_map, conditions_list)
    tags_vec_dic, tags_vec_list = sentence_to_avg(word_to_vec_map, tags_list)
    symptoms_vec_dic, symptoms_vec_list = sentence_to_avg(word_to_vec_map, symptoms_list)

    # plot_Kmeans_error(conditions_vec_list)
    # plot_Kmeans_error(tags_vec_list)
    # plot_Kmeans_error(symptoms_vec_list)

    condition_cluster = get_clusters(conditions_vec_list, conditions_vec_dic, n_condition_clusters)
    print(condition_cluster)
    tag_cluster = get_clusters(tags_vec_list, tags_vec_dic, n_tag_clusters)
    print(tag_cluster)
    symptom_cluster = get_clusters(symptoms_vec_list, symptoms_vec_dic, n_symptom_clusters)
    print(symptom_cluster)

    write_dic(countries,condition_cluster,tag_cluster,symptom_cluster)


#----------------------------------------------------------------------------

def read_dic():    #read in the dic for conditions, tags, and symptoms
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/dic.txt",'r',encoding='utf-8') as fin:
            Dic={"countries_dic":{"":0},"conditions_dic":{},"tags_dic":{},"symptoms_dic":{}}

            dic_selected = ""
            for line in fin:
                record=line.strip().split(',')


                if(record[0]=="COUNTRY"):
                    dic_selected="countries_dic"

                elif (record[0] == "CONDITION"):
                    dic_selected = "conditions_dic"

                elif(record[0]=="TAG"):
                    dic_selected = "tags_dic"

                elif(record[0]=="SYMPTOM"):
                    dic_selected = "symptoms_dic"

                else:
                    Dic[dic_selected][record[0]]=int(record[1])

            return Dic


# ----------------------------------------------------------------------------
def data_in_dic():  #keep only the data that is in the dic
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/seperate_groups.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/data_in_dic.txt",'w',encoding="utf-8") as fout:
            allDic = read_dic()

            conditions_dic = allDic["conditions_dic"]
            tags_dic = allDic["tags_dic"]
            symptoms_dic = allDic["symptoms_dic"]
            for line in fin:
                flag=True
                record=line.strip().split(',')
                if(record[0]=="Condition"):
                    if(not record[1] in conditions_dic):
                        flag=False

                elif(record[0]=="Tag"):
                    if(not record[1] in tags_dic):
                        flag=False

                elif(record[0]=="Symptom"):
                    if(not record[1] in symptoms_dic):
                        flag=False

                if(flag):
                    fout.write(line)


#----------------------------------------------------------------------------
def keep_good_data():
    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/data_in_dic.txt", 'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/keep_good_data.txt",'w',encoding="utf-8") as fout:
            module = []
            for line in fin:
                if ("==" in line):
                    flag=True
                    symptom_counter=0
                    for row in module:
                        record=row.split(',')
                        if(record[0]=="Symptom"):
                            symptom_counter+=1

                    if(len(module)*0.3<symptom_counter):
                        flag=False
                    if (len(module)<15):
                         flag=False

                    if(flag):
                        for i in module:
                            fout.write(i)


                    module = []
                    module.append(line)
                else:
                    module.append(line)



#----------------------------------------------------------------------------
def convert_to_number():  #convert all strings to numbers for machine learning
    gender_dic={"male":0, "female":1,"other":2}
    allDic=read_dic()
    countries_dic=allDic["countries_dic"]
    conditions_dic=allDic["conditions_dic"]
    tags_dic=allDic["tags_dic"]
    symptoms_dic=allDic["symptoms_dic"]
    country_size=countries_dic[max(countries_dic,key=countries_dic.get)]

    condition_size = conditions_dic[max(conditions_dic, key=conditions_dic.get)]+1
    tag_size = tags_dic[max(tags_dic, key=tags_dic.get)]+1
    symptom_size = symptoms_dic[max(symptoms_dic, key=symptoms_dic.get)]+1

    with open("/Users/liuliu/myDocuments/flare_down/code/Version6/keep_good_data.txt",'r',encoding="utf-8") as fin:
        with open("/Users/liuliu/myDocuments/flare_down/code/Version6/training.txt", 'w') as fout1:
            with open("/Users/liuliu/myDocuments/flare_down/code/Version6/dev.txt", 'w') as fout2:
                s=""
                countries_value=[0]*country_size
                conditons_value=[0]*condition_size
                tags_value=[0]*tag_size
                gender_value=[0]*3
                symptoms_value=[0]*symptom_size
                counter=0

                first=True
                for line in fin:

                    fout=fout1


                    record=line.split(',')
                    if("==" in record[0] ):
                        for i in range(len(conditons_value)):

                            s += (str(conditons_value[i]) + ',')
                        for i in range(len(tags_value)):

                            s += (str(tags_value[i]) + ',')
                        for i in range(len(symptoms_value)):

                            s += (str(symptoms_value[i]) + ',')

                        if(not first):
                            fout.write(s+'\n')
                        else:
                            first=False


                        s = ""
                        conditons_value = [0] * condition_size
                        tags_value = [0] * tag_size
                        symptoms_value = [0]*symptom_size

                        counter+=1

                       # s+=record[0]
                        for i in range(1,4):

                            if(i==1):
                                if(record[i]!=''):

                                    s+=(str(record[1])+',')
                                else:
                                    s+=("0,")

                            if(i==2):
                                if (record[i] != '' and record[i]!="doesnt_say"):

                                    gender_value[gender_dic[record[2]]]=1

                                for j in gender_value:

                                    s += (str(j) + ',')


                            if(i==3):
                                if (record[i] != ''):
                                    countries_value[countries_dic[record[3]]-1] = 1


                                for j in range(len(countries_value)):

                                    s += (str(countries_value[j]) + ',')

                        gender_value=[0]*3
                        countries_value=[0]*country_size

                    elif(record[0]=="Condition"):

                        conditons_value[conditions_dic[record[1]]]=int(record[-2])


                    elif(record[0]=="Tag"):

                        tags_value[tags_dic[record[1]]]=1


                    elif(record[0]=="Symptom"):

                        symptoms_value[symptoms_dic[record[1]]]=1

                print("number of samples",counter)

    return symptom_size,country_size+condition_size+tag_size+symptom_size+4

#----------------------------------------------------------------------------


def read_set(File,symptom_size,overall_size):
    X=[]
    Y=[]
    with open(File, 'r') as fin:
        for line in fin:
            record=line.split(',')

            tmp=[]

            for i in range(overall_size-symptom_size):
                tmp.append(int(record[i]))

            X.append(tmp)

            tmp=[]
            for i in range(overall_size-symptom_size,overall_size):

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





def feature_scaling(X_train,X_test):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    return X_train,X_test





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

    precision =0
    recall =0
    harmonic_mean=0
    if(true_pos==0):
        precision = 0
        recall = 0
        harmonic_mean=0
    else:
        precision=true_pos/(true_pos+fal_pos)
        recall=true_pos/(true_pos+fal_neg)
        harmonic_mean=2*precision*recall/(precision+recall)
    return precision,recall,harmonic_mean


def tuning_hyper_parameters():

    hyper_learning_rate=[0.0003,0.001,0.003,0.009,0.01,0.03,0.09,0.1,0.3]
    hyper_alpha=[1,2,4,6,9,]
    hyper_n_units=range(50,501,50)

    hyper_threshold=[0.4,0.35,0.3,0.25]
    mean_curve_train = []
    mean_curve_test = []


    clf = MLPClassifier(solver='adam',alpha=1.4, hidden_layer_sizes=(300), random_state=1,max_iter=1000,learning_rate_init=0.003)
    clf.fit(X_train, Y_train)
    precision, recall, harmonic_mean = my_score(clf.predict(X_train), Y_train)
    performance_train=harmonic_mean
    # mean_curve_train.append(harmonic_mean)
    print("train: ", precision, recall, harmonic_mean)
    precision, recall, harmonic_mean = my_score(clf.predict(X_test), Y_test)
    performance_test=harmonic_mean
    # mean_curve_test.append(harmonic_mean)
    print("test: ", precision, recall, harmonic_mean)

    return performance_train, performance_test

    # plt.scatter(hyper_learning_rate,mean_curve_train)
    # plt.scatter(hyper_learning_rate,mean_curve_test)
    # plt.xlabel("learning rate")
    # plt.ylabel("performance")
    # plt.title("performance against learning rate")
    # plt.show()

    # for j in hyper_alpha:
    #     mean_curve_train = []
    #     mean_curve_test = []
    #     print("alpha=",j)
    #     clf = MLPClassifier(solver='adam', alpha=j, hidden_layer_sizes=(50), random_state=1, max_iter=1,
    #                         learning_rate_init=0.001, warm_start=True)
    #
    #     for i in range(200):
    #
    #         print("number of iterations=:", i)
    #
    #
    #         clf.fit(X_train, Y_train)
    #
    #         #print(clf.predict_proba(X_train[0].reshape(1, -1)))
    #         #print(clf.predict(X_train[0].reshape(1, -1)))
    #         precision, recall, harmonic_mean = my_score(clf.predict(X_train), Y_train)
    #         mean_curve_train.append(harmonic_mean)
    #         print("train: ", precision, recall, harmonic_mean)
    #         precision, recall, harmonic_mean = my_score(clf.predict(X_test), Y_test)
    #         mean_curve_test.append(harmonic_mean)
    #         print("test: ", precision, recall, harmonic_mean)
    #
    #
    #
    #     # plt.plot(clf.loss_curve_)
    #     plt.plot(mean_curve_train)
    #     plt.plot(mean_curve_test)
    #     plt.show()

i=40
j=40
k=50


print("n_condition_clusters=",i,"n_tag_clusters=",j,"n_symptom_clusters=",k)
read_dic()
data_in_dic()
keep_good_data()
symptom_size,overall_size=convert_to_number()

X,Y,m=read_set("/Users/liuliu/myDocuments/flare_down/code/Version6/training.txt",symptom_size,overall_size)

kf = KFold(n_splits=14,shuffle=True)

avg_performance_train = 0
avg_performance_test=0
for train_index, test_index in kf.split(X):


    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    print(Y_train.shape[0],Y_test.shape[0])

    X_train,X_test=feature_scaling(X_train.astype(float),X_test.astype(float))

    a,b=tuning_hyper_parameters()
    avg_performance_train+=a
    avg_performance_test+=b

avg_performance_test/=14
avg_performance_train/=14
print(avg_performance_train,avg_performance_test)