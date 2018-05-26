
mostFrequentSymptoms=("fatigue","muscle","headache","joint pain","back pain")
mostFrequentConditions=("chronic fatigue syndrome","fibromyalgia","fatigue","anxiety","disorder","depression")
mostFrequentTreatment=("vitamin","magnesium","ibuprofen","tramadol","exercise")
mostFrequentTag=("tired","sleep","stressed","walked")

#----------------------------------------------------------------------------
def delete_dateAndtrackableID():
    with open ("/Users/liuliu/Desktop/Python/flare_down/Version1/txtFileOfCsv.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/delete_dateAndtrackableID.txt",'w') as fout:
            for line in fin:
                counter=0
                for s in line.split(','):
                    if(counter!=4 and counter!=5):

                        fout.write(s)
                        if(s!='\n'):
                            fout.write(',')
                    counter+=1

# ----------------------------------------------------------------------------
def delete_dateAndtrackableID2():
    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/txtFileOfCsv.txt", 'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/delete_dateAndtrackableID.txt", 'w') as fout:
            for line in fin:
                record=line.split(',')
                for i in range(len(record)):
                    if (i!= 4 and i != 5):

                        fout.write(record[i])
                        if (record[i] != '\n'):
                            fout.write(',')

#----------------------------------------------------------------------------
def rearrange():           #rearrange data such that all information under the same user will be grouped
    with open ("/Users/liuliu/Desktop/Python/flare_down/Version1/delete_dateAndtrackableID.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/rearrange.txt",'w') as fout:
            preID = ""
            for line in fin:

                record=line.split(',')


                if(record[0]!=preID):
                    preID=record[0]
                    fout.write('\n')

                    for i in range(4):
                        fout.write(record[i])

                        if(record[i]!='\n'):
                            fout.write(',')
                    fout.write('\n')

                for i in range(4,len(record)):
                    fout.write(record[i])
                    if (record[i] != '\n'):
                        fout.write(',')



#----------------------------------------------------------------------------
def discard_low_frequency_features():


    with open ("/Users/liuliu/Desktop/Python/flare_down/Version1/rearrange.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/discard_less_frequent_features.txt",'w') as fout:
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

    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/discard_less_frequent_features.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/remove_incomplete_data.txt",'w') as fout:
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
def deal_with_featuresBeforeSymptom(till_symptom,fout):  #keep only two most frequent conditions
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


def twoFrequentConditionsBeforeSymptom():
    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/remove_incomplete_data.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/twoFrequentFeaturesBeforeSymptom.txt",'w') as fout:

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

def allFrequentFeaturesBeforeSymptom():
    # keep all conditions that are in the tuple:mostFrequentConditions while only keep one most frequent tag
    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/remove_incomplete_data.txt", 'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/allFrequentConditionsBeforeSymptom.txt", 'w') as fout:

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

def generateCountries():
    #generate a file that contains countries and their mappings
    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/allFrequentConditionsBeforeSymptom.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/countries.txt",'w') as fout:
            countries=[]
            for line in fin:
                if("==" in line):
                    tmp=line.split(',')
                    if(not tmp[3] in countries):
                        countries.append(tmp[3])

            for i in range(len(countries)):
                fout.write(countries[i])
                fout.write(' ')
                fout.write(str(i+1))
                fout.write('\n')

#----------------------------------------------------------------------------
def convert_to_number():
    #map all text data to number data
    gender={"doesnt_say":0,"male":1, "female":2,"other":3}
    countries={"us":1,"gb":2,"ca":3,"za":4,"fi":5,"au":6,"ch":7,"pt":8,"be":9,"de":10,"sg":11,"il":12,"is":13,"nz":14,"ua":15,"hk":16,"dk":17,"uy":18,"ie":19,"es":20,"dz":21}
    with open("/Users/liuliu/Desktop/Python/flare_down/Version1/twoFrequentFeaturesBeforeSymptom.txt",'r') as fin:
        with open("/Users/liuliu/Desktop/Python/flare_down/Version1/training.txt", 'w') as fout:
            s=""
            conditons=[0]*6
            tagNum=0
            symNum=0
            for line in fin:
                record=line.split(',')
                if("==" in record[0] ):
                    for i in range(6):
                        s += (str(conditons[i]) + ',')
                    s += (str(tagNum) + ',')
                    s += (str(symNum) + ',')
                    fout.write(s+'\n')


                    s=""
                   # s+=record[0]
                    s+=(str(record[1])+',')
                    s+=(str(gender[record[2]])+',')
                    s+=(str(countries[record[3]])+',')

                elif(record[0]=="condition"):
                    for i in range(len(mostFrequentConditions)):
                        if(mostFrequentConditions[i] in record[1]):
                            conditons[i]=record[2]
                            break

                elif(record[0]=="tag"):
                    for i in range(len(mostFrequentTag)):
                        if(mostFrequentTag[i] in record[1]):
                            tagNum=i+1
                            break

                elif(record[0]=="symptom"):
                    for i in range(len(mostFrequentSymptoms)):
                        if(mostFrequentSymptoms[i] in record[1]):
                            symNum=i+1
                            break


#----------------------------------------------------------------------------

#generateCountries()
# discard_low_frequency_features()
# remove_incomplete_data()
twoFrequentConditionsBeforeSymptom()
convert_to_number()

