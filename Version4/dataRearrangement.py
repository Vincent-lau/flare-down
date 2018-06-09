from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import codecs
from gensim.models.keyedvectors import KeyedVectors




# ----------------------------------------------------------------------------
def delete_trackableID():    #delete the column trackableID
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/txtFileOfCsv.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/delete_trackableID.txt", 'w') as fout:
            for line in fin:
                record=line.split(',')
                for i in range(len(record)):
                    if (i != 5):

                        fout.write(record[i])
                        if (record[i] != '\n'):
                            fout.write(',')


#----------------------------------------------------------------------------
def rearrange():           #rearrange data such that all information under the same user will be grouped
    with open ("/Users/liuliu/My Documents/flare_down/code/Version4/delete_trackableID.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/rearrange.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/sort_by_date.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/sort_by_date.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/discard_noSymptoms.txt", 'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/discard_noSymptoms.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/discard_treatment_weather_zeroSymptom.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/discard_treatment_weather_zeroSymptom.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/seperate_groups.txt",'w') as fout:
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

#----------------------------------------------------------------------------

def read_dic():    #read in the dic for conditions, tags, and symptoms
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/dic.txt",'r') as fin:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/seperate_groups.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/data_in_dic.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/data_in_dic.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/keep_good_data.txt",'w') as fout:
            module = []
            for line in fin:
                if ("==" in line):
                    flag=True
                    symptom_counter=0
                    for row in module:
                        record=row.split(',')
                        if(record[0]=="Symptom"):
                            symptom_counter+=1

                    if(len(module)*0.4<symptom_counter):
                        flag=False
                    if (len(module)<8):
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


    with open("/Users/liuliu/My Documents/flare_down/code/Version4/keep_good_data.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version4/training.txt", 'w') as fout1:
            with open("/Users/liuliu/My Documents/flare_down/code/Version4/dev.txt", 'w') as fout2:
                s=""
                countries_value=[0]*36
                conditons_value=[0]*19
                tags_value=[0]*18
                gender_value=[0]*3
                symptoms_value=[0]*19
                counter=0

                first=True
                for line in fin:
                    if(counter<2146-400):

                        fout=fout1
                    else:
                        fout=fout2


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
                        conditons_value = [0] * 19
                        tags_value = [0] * 18
                        symptoms_value = [0]*19

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
                                    countries_value[countries_dic[record[3]]] = 1


                                for j in range(len(countries_value)):

                                    s += (str(countries_value[j]) + ',')

                        gender_value=[0]*3
                        countries_value=[0]*36

                    elif(record[0]=="Condition"):

                        conditons_value[conditions_dic[record[1]]]=int(record[-2])


                    elif(record[0]=="Tag"):

                        tags_value[tags_dic[record[1]]]=1


                    elif(record[0]=="Symptom"):

                        symptoms_value[symptoms_dic[record[1]]]=1


keep_good_data()
convert_to_number()
#----------------------------------------------------------------------------
def read_word_vecs(word_vecs):

    with codecs.open(word_vecs, 'r',encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        first=True
        for line in f:
            if(first):
                first=False
                continue

            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

# words, word_to_vec_map = read_word_vecs("/Users/liuliu/Downloads/PMC-w2v.txt")

#----------------------------------------------------------------------------
def generate_list():    #generate lists that has all the conditions, tags and symptoms in them for the conversion to averaged word vectors
    with open("/Users/liuliu/My Documents/flare_down/code/Version4/sort_by_date.txt",'r') as fin:

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
        return conditions_list,tags_list,symptoms_list
#----------------------------------------------------------------------------

def sentence_to_avg(word_to_vec_map,X):     #convert a particular string of condition to a word vector by averaging every word in the string
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


# conditions_list,tags_list,symptoms_list=generate_list()
#
# conditions_vec_dic,conditions_vec_list=sentence_to_avg(word_to_vec_map,conditions_list)
# tags_vec_dic,tags_vec_list=sentence_to_avg(word_to_vec_map,tags_list)
# symptoms_vec_dic,symptoms_vec_list=sentence_to_avg(word_to_vec_map,symptoms_list)
#
#----------------------------------------------------------------------------

def visualisation(x1):    #applying PCA to visualise the data

    pca = PCA(n_components=2)
    newX1 = pca.fit_transform(x1)
    print("PCA variance sum of word vectors",np.sum(pca.explained_variance_ratio_))


    plt.scatter(newX1[:,0],newX1[:,1],c="red")

    plt.show()



#----------------------------------------------------------------------------


def plot_Kmeans_error(X):     #plot k-means error to choose number of centroids
    error=[]
    for i in range(4,30):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        print(kmeans.inertia_,i)
        error.append([i,kmeans.inertia_])
       # if(i==16 or i==14):
        #    visualisation(X,kmeans.cluster_centers_)


    error=np.array(error)
    plt.scatter(error[:,0],error[:,1])
    plt.show()

#----------------------------------------------------------------------------

def print_clusters(X_list,X_dic,n_centroids):    #print every entry in the list and its corresponding centroid
    X_list=np.array(X_list)
    kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(X_list)
    clusters=[]

    for i in range(len(X_dic.keys())):
        clusters.append([list(X_dic.keys())[i], kmeans.predict(X_list[i].reshape(1,-1))])

    clusters.sort(key= lambda clusters:clusters[1])
    for row in clusters:
        print(row)



