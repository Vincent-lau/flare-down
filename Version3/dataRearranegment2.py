from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import codecs
from gensim.models.keyedvectors import KeyedVectors



# ----------------------------------------------------------------------------
def delete_trackableID():
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/txtFileOfCsv.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/delete_trackableID.txt", 'w') as fout:
            for line in fin:
                record=line.split(',')
                for i in range(len(record)):
                    if (i != 5):

                        fout.write(record[i])
                        if (record[i] != '\n'):
                            fout.write(',')


#----------------------------------------------------------------------------
def rearrange():           #rearrange data such that all information under the same user will be grouped
    with open ("/Users/liuliu/My Documents/flare_down/code/Version3/delete_trackableID.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/rearrange.txt",'w') as fout:
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

def sort_by_date():             #sort every group according to dates and delete all date afterwards
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/sort_by_date.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/sort_by_date.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/discard_noSymptoms.txt", 'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/discard_noSymptoms.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/discard_treatment_weather_zeroSymptom.txt",'w') as fout:
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
def select_symptom():  #choose the symptom with the highest number among consecutive symptoms
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/discard_treatment_weather_zeroSymptom.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/select_symptom.txt",'w') as fout:
            symptoms=[]
            for line in fin:
                record=line.split(',')
                if(record[0]!="Symptom"):
                    if(len(symptoms)):
                        symptoms.sort(key=lambda symptoms: int(symptoms[-2]),reverse=True)
                        for i in symptoms[0]:
                            fout.write(i)
                            if(i!='\n'):
                                fout.write(',')
                        symptoms=[]
                    fout.write(line)

                else:
                    symptoms.append(record)


#----------------------------------------------------------------------------
def discard_low_frequency_features():


    with open ("/Users/liuliu/My Documents/flare_down/code/Version3/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/discard_less_frequent_features.txt",'w') as fout:
            for line in fin:

                line=line.lower()
                line=line.split(',')
                flag=True
                if(line[0]=="symptom"):
                    for i in mostFrequentSymptoms:
                        if(i in line[1]):
                            break
                    else:
                        flag=False

                elif(line[0]=="condition"):
                    for i in mostFrequentConditions:
                        if(i in line[1]):
                            break
                    else:
                        flag=False



                elif (line[0] == "tag"):
                    for i in mostFrequentTag:
                        if (i in line[1]):
                            break
                    else:
                        flag = False


                elif("==" in line[0] or line[0]=='\n'):
                    flag=True
                else:
                    flag=False
                if(flag):
                    for i in line:
                        fout.write(i)
                        if(i !='\n'):
                            fout.write(',')


#----------------------------------------------------------------------------
def remove_incomplete_data():

    with open("/Users/liuliu/Desktop/Python//Users/liuliu/My Documents/flare_down/code/Version3/discard_less_frequent_features.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/remove_incomplete_data.txt",'w') as fout:
            module=[""]
            for line in fin:
                flag=True
                if("==" in line):

                    user_information=module[0].split(',')
                    for i in  range(1,len(user_information)):
                        if(user_information[i]==''):
                            flag=False


                    if(flag):
                        for i in module:
                            fout.write(i)
                        fout.write('\n')

                    module=[[]]
                    module[0]=line
                else:
                    module.append(line)



#----------------------------------------------------------------------------
def seperate_groups():  #seperate groups such that every group ends with a single symptom and discard those without a symptom at the end
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/select_symptom.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/seperate_groups.txt",'w') as fout:
            module = []
            for line in fin:
                if ("==" in line):
                    till_symptom = []
                    for row in module:

                        record= row.split(',')
                        if (record[0] == "Symptom"):



                            fout.write(module[0])
                            for i in till_symptom:
                                fout.write(i)
                            fout.write(row)
                            fout.write('\n')
                            till_symptom.clear()

                        elif(not "==" in row):
                            till_symptom.append(row)

                    module.clear()
                    till_symptom.clear()
                    module.append(line)

                elif (line != ''):
                    module.append(line)


#----------------------------------------------------------------------------

def generate_list():
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/sort_by_date.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/countries.txt",'w') as fout:
            countries={"":0}
            tags_list=[]
            symptoms_list=[]
            conditions_list=[]
            counter_conunrties=0
            counter_conditions = -1
            counter_tags=-1
            counter_symptoms=0
            for line in fin:
                record=line.split(',')
                if("==" in line):
                    if(not record[3] in countries):
                        counter_conunrties+=1
                        countries[record[3]]=counter_conunrties

                elif (record[0] == "Condition"):

                    if (not record[1] in conditions_list):

                        conditions_list.append(record[1])

                elif(record[0]=="Tag"):

                    if(not record[1] in tags_list):

                        tags_list.append(record[1])

                elif(record[0]=="Symptom"):
                    if(not record[1] in symptoms_list):

                        symptoms_list.append(record[1])

            for i in countries:
                fout.write(i)
                fout.write('\n')
            fout.write('\n')

            counter=0
            for i in conditions_list:
                counter+=1
                fout.write(i+' '+str(counter)+'\n')


            for i in tags_list:
                fout.write(i)
                fout.write('\n')

            fout.write('\n')

            for i in symptoms_list:
                fout.write(i)
                fout.write('\n')

            fout.write('\n')

            return conditions_list,tags_list,symptoms_list


#----------------------------------------------------------------------------
def convert_to_number():
    gender={"doesnt_say":0,"male":1, "female":2,"other":3}
    countries,conditions,tags,symptoms=generateDic()
    with open("/Users/liuliu/My Documents/flare_down/code/Version3/seperate_groups.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version3/training.txt", 'w') as fout1:
            with open("/Users/liuliu/My Documents/flare_down/code/Version3/test.txt", 'w') as fout2:
                s=""
                conditons_value=[0]*len(conditions)
                tags_value=[0]*len(tags)
                symptoms_value=0
                counter=0

                first=True
                for line in fin:
                    if(0<=counter<2000):
                        fout=fout1
                    elif(2000<=counter<2400):
                        fout=fout2
                    else:
                        break
                    record=line.split(',')
                    if("==" in record[0] ):
                        for i in range(len(conditons_value)):
                            s += (str(conditons_value[i]) + ',')
                        for i in range(len(tags_value)):
                            s += (str(tags_value[i]) + ',')

                        s += (str(symptoms_value) + ',')

                        if(not first):
                            fout.write(s+'\n')
                        else:
                            first=False

                        s = ""
                        conditons_value = [0] * len(conditions)
                        tags_value = [0] * len(tags)
                        symptoms_value = 0
                        counter+=1

                       # s+=record[0]
                        for i in range(1,4):
                            if(record[i]==''):
                                s+=(str(0)+',')
                                continue
                            if(i==1):
                                s+=(str(record[1])+',')
                            if(i==2):
                                s+=(str(gender[record[2]])+',')
                            if(i==3):
                                s+=(str(countries[record[3]])+',')

                    elif(record[0]=="Condition"):

                        conditons_value[conditions[record[1]]]=int(record[-2])


                    elif(record[0]=="Tag"):
                        tags_value[tags[record[1]]]=1


                    elif(record[0]=="Symptom"):
                        symptoms_value=symptoms[record[1]]


#----------------------------------------------------------------------------


def read_glove_vecs(glove_file):

    with codecs.open(glove_file, 'r',encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:

            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs("/Users/liuliu/Downloads/glove.6B/glove.6B.50d.txt")
print("---------------------word vectors loaded------------------------")



def sentence_to_avg(word_to_vec_map,X):
    X_vec_dic={}
    X_vec_list=[]


    for sentence in X:
        words = sentence.lower().split(' ')
        avg =0
        for w in words:
            if(not w in word_to_vec_map):

                break
            else:
                avg += word_to_vec_map[w]

        else:
            avg /= len(words)
            X_vec_dic[sentence]=avg
            X_vec_list.append(avg)


    return X_vec_dic,X_vec_list

conditions_list,tags_list,symptoms_list=generate_list()

conditions_vec_dic,conditions_vec_list=sentence_to_avg(word_to_vec_map,conditions_list)
tags_vec_dic,tags_vec_list=sentence_to_avg(word_to_vec_map,tags_list)
symptoms_vec_dic,symptoms_vec_list=sentence_to_avg(word_to_vec_map,symptoms_list)

print("-------------------sentence averaged-------------------")

def visualisation(x1,x2=None):

    pca = PCA(n_components=2)
    newX1 = pca.fit_transform(x1)
    print("PCA variance sum of word vectors",np.sum(pca.explained_variance_ratio_))
    #
    # if(x2!=None):
    #     newX2=pca.fit_transform(x2)
    #     print("PCA variance sum of centroids",np.sum(pca.explained_variance_ratio_))
    #     plt.scatter(newX2[:, 0], newX2[:, 1], c="blue")


    plt.scatter(newX1[:,0],newX1[:,1],c="red")

    plt.show()






def plot_Kmeans_error(X):
    error=[]
    for i in range(4,40):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        print(kmeans.inertia_,i)
        error.append([i,kmeans.inertia_])
        if(i==22):
            visualisation(X,kmeans.cluster_centers_)


    error=np.array(error)
    plt.scatter(error[:,0],error[:,1])
    plt.show()


# plot_Kmeans_error(tags_vec_list)
# plot_Kmeans_error(symptoms_vec_list)

def print_clusters(X_list,X_dic,n_centroids):
    X_list=np.array(X_list)
    kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(X_list)
    clusters=[]

    for i in range(len(X_dic.keys())):
        clusters.append([list(X_dic.keys())[i], kmeans.predict(X_list[i].reshape(1,-1))])

    clusters.sort(key= lambda clusters:clusters[1])
    for row in clusters:
        print(row)


print_clusters(symptoms_vec_list,symptoms_vec_dic,22)