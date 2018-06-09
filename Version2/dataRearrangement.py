
mostFrequentSymptoms=("fatigue","muscle","headache","joint pain","back pain")
mostFrequentConditions=("chronic fatigue syndrome","fibromyalgia","fatigue","anxiety","disorder","depression")
mostFrequentTreatment=("vitamin","magnesium","ibuprofen","tramadol","exercise")
mostFrequentTag=("tired","sleep","stressed","walked")


# ----------------------------------------------------------------------------
def delete_trackableID():
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/txtFileOfCsv.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/delete_trackableID.txt", 'w') as fout:
            for line in fin:
                record=line.split(',')
                for i in range(len(record)):
                    if (i != 5):

                        fout.write(record[i])
                        if (record[i] != '\n'):
                            fout.write(',')


#----------------------------------------------------------------------------
def rearrange():           #rearrange data such that all information under the same user will be grouped
    with open ("/Users/liuliu/My Documents/flare_down/code/Version2/delete_trackableID.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/rearrange.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/sort_by_date.txt",'w') as fout:
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


#----------------------------------------------------------------------------
def get_allConditions():
    conditions={}
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/rearrange.txt",'r') as fin:
        for line in fin:
            record=line.split(',')
            if(record[0]=="Condition"):
                if(record[1] in conditions):
                    conditions[record[1]]+=1
                else:
                    conditions[record[1]]=1
    for i in conditions:
        print(i,conditions[i])

# ----------------------------------------------------------------------------
def discard_noSymptoms():   #discard groups with no symptoms i.e. no y value
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/sort_by_date.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/discard_noSymptoms.txt", 'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/discard_noSymptoms.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/discard_treatment_weather_zeroSymptom.txt",'w') as fout:
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
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/discard_treatment_weather_zeroSymptom.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/select_symptom.txt",'w') as fout:
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
def discard_low_frequency_features():  #those features that are not in the tuple will be discarded


    with open ("/Users/liuliu/My Documents/flare_down/code/Version2/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/discard_less_frequent_features.txt",'w') as fout:
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

    with open("/Users/liuliu/Desktop/Python//Users/liuliu/My Documents/flare_down/code/Version2/discard_less_frequent_features.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/remove_incomplete_data.txt",'w') as fout:
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
def deal_with_featuresBeforeSymptom(till_symptom,fout):
    if(len(till_symptom) == 0):
        return (False,[])



    if("==" in till_symptom[0]):
        till_symptom[1:]=sorted(till_symptom[1:])
    else:
        till_symptom.sort()


   # print("till_symptom",till_symptom)
    conditionsKept=[]
    tagKept=[]
    othersKept=[]

    for record in till_symptom:
        record1=record.split(',')
        if("==" in record1[0]):

            continue

        elif(record1[0]=="condition"):
            for i in range(len(mostFrequentConditions)):
                if(mostFrequentConditions[i] in record1[1]):

                    conditionsKept.append((record,i))

                    break
        elif(record1[0]=="tag"):
            for i in range(len(mostFrequentTag)):
                if(mostFrequentTag[i] in record1[1]):

                    tagKept.append((record,i))

                    break
        else:

            othersKept.append((record,100))




    flag_condition=False
    if (len(conditionsKept)>=2):

        conditionsKept.sort(key=lambda conditionsKept: conditionsKept[1])

        while(len(conditionsKept)>2):
            conditionsKept.pop()
        flag_condition=True



    flag_tag=False
    if(len(tagKept)>=2):
        flag_tag=True
        tagKept.sort(key=lambda tagKept: tagKept[1])
        while(len(tagKept)>2):
            tagKept.pop()

    flag=flag_condition and flag_tag

    kept=conditionsKept+tagKept+othersKept
    return (flag,kept)
    # for i in till_symptom:
    #     fout.write(i)


def twoFrequentFeaturesBeforeSymptom():
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/remove_incomplete_data.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/twoFrequentFeaturesBeforeSymptom.txt",'w') as fout:

            #keep the most frequent two, if not exist, set flag=false
            module=[""]
            for line in fin:
                if("==" in line):
                    till_symptom = []
                    for record in module:

                        tmp = record.split(',')
                        if (tmp[0] == "symptom"):

                            flag,kept=deal_with_featuresBeforeSymptom(till_symptom,fout)


                            if(flag):
                                fout.write(module[0])
                                for j in kept:
                                    fout.write(j[0])
                                fout.write(record)
                                fout.write('\n')
                            till_symptom=[]
                        else:
                            till_symptom.append(record)

                    module=[]

                    module.append(line)
                elif(line!=''):
                    module.append(line)


#----------------------------------------------------------------------------
def seperate_groups():  #seperate groups such that every group ends with a single symptom and discard those without a symptom at the end
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/select_symptom.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/seperate_groups.txt",'w') as fout:
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
def deal_with_featuresBeforeSymptom2(till_symptom,fout):
    if(len(till_symptom) == 0):
        return (False,[])



    if("==" in till_symptom[0]):
        till_symptom[1:]=sorted(till_symptom[1:])
    else:
        till_symptom.sort()


   # print("till_symptom",till_symptom)
    conditionsKept=[]
    tagKept=[]
    othersKept=[]

    for record in till_symptom:
        record1=record.split(',')
        if("==" in record1[0]):

            continue

        elif(record1[0]=="condition"):
            for i in range(len(mostFrequentConditions)):
                if(mostFrequentConditions[i] in record1[1]):

                    conditionsKept.append((record,i))

                    break
        elif(record1[0]=="tag"):
            for i in range(len(mostFrequentTag)):
                if(mostFrequentTag[i] in record1[1]):

                    tagKept.append((record,i))

                    break
        else:

            othersKept.append((record,100))




    # flag_condition=False
    # if (len(conditionsKept)>=2):
    #
    #     conditionsKept.sort(key=lambda conditionsKept: conditionsKept[1])
    #
    #     while(len(conditionsKept)>2):
    #         conditionsKept.pop()
    #     flag_condition=True
    #
    #
    #
    flag_tag=False
    if(len(tagKept)>=1):
        flag_tag=True
        tagKept.sort(key=lambda tagKept: tagKept[1])
        while(len(tagKept)>1):
            tagKept.pop()

    flag=flag_tag


    kept=conditionsKept+tagKept+othersKept
    return (flag,kept)


def allFrequentConditionsBeforeSymptom():
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/remove_incomplete_data.txt", 'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/allFrequentConditionsBeforeSymptom.txt", 'w') as fout:

            module = [""]
            for line in fin:
                if ("==" in line):
                    till_symptom = []
                    for record in module:

                        tmp = record.split(',')
                        if (tmp[0] == "symptom"):

                            flag, kept = deal_with_featuresBeforeSymptom2(till_symptom, fout)

                            if (flag):
                                fout.write(module[0])
                                for j in kept:
                                    fout.write(j[0])
                                fout.write(record)
                                fout.write('\n')
                            till_symptom = []
                        else:
                            till_symptom.append(record)

                    module = []

                    module.append(line)
                elif (line != ''):
                    module.append(line)
#----------------------------------------------------------------------------

def generateDic():
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/seperate_groups.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/countries.txt",'w') as fout:
            countries={"":0}
            conditions = {}
            tags={}
            symptoms={}
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

                    if (not record[1] in conditions):
                        counter_conditions += 1
                        conditions[record[1]] = counter_conditions

                elif(record[0]=="Tag"):

                    if(not record[1] in tags):
                        counter_tags+=1
                        tags[record[1]]=counter_tags

                elif(record[0]=="Symptom"):
                    if(not record[1] in symptoms):
                        counter_symptoms+=1
                        symptoms[record[1]]=counter_symptoms

            for i in countries:
                fout.write(i)
                fout.write('\n')
            fout.write('\n')

            counter=0
            for i in conditions:
                counter+=1
                fout.write(i+' '+str(counter)+'\n')


            for i in tags:
                fout.write(i)
                fout.write('\n')

            fout.write('\n')

            for i in symptoms:
                fout.write(i)
                fout.write('\n')

            fout.write('\n')
            print(len(symptoms))
            return countries,conditions,tags,symptoms


#----------------------------------------------------------------------------
def convert_to_number():
    gender={"doesnt_say":0,"male":1, "female":2,"other":3}
    countries,conditions,tags,symptoms=generateDic()
    with open("/Users/liuliu/My Documents/flare_down/code/Version2/seperate_groups.txt",'r') as fin:
        with open("/Users/liuliu/My Documents/flare_down/code/Version2/training.txt", 'w') as fout1:
            with open("/Users/liuliu/My Documents/flare_down/code/Version2/dev.txt", 'w') as fout2:
                s=""
                conditons_value=[0]*len(conditions)
                tags_value=[0]*len(tags)
                symptoms_value=0
                counter=0

                first=True
                for line in fin:
                    if(0<=counter<1000):
                        fout=fout1
                    elif(1000<=counter<1400):
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

convert_to_number()
#----------------------------------------------------------------------------



# discard_low_frequency_features()
# remove_incomplete_data()
# allFrequentConditionsBeforeSymptom()
# convert_to_number()

