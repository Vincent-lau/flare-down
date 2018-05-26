import csv

#----------------------------------------------------------------------
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(utf_8_encoder(file_obj))
    dic={}
    for row in reader:

        yield [unicode(cell, 'utf-8') for cell in row]




with open("fd-export",'r') as f1:


    with open('fd-export2', 'w') as f:

     #   spamwriter = csv.writer(csvfile1, delimiter=',',
      #                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
        counter=0
        pre = ""

        for row in f1:
            f.write(row)

            # if(row[0]!=pre):
            #     f.write('\n')
            #     pre = row[0]
            #
            #     for i in range(6):
            #
            #         f.write(row[i])
            #         f.write(' ')
            # else:
            #     for i in range(4,7):
            #         f.write(row[i])
            #         f.write(' ')
            #
            # f.write("\n")

               # print(pre)

            counter+=1
            if(counter>1000):
                break


#csv_path1 = "/Users/liuliu/Desktop/Octave/crest/fd-export1.csv"
#with open(csv_path1,'w',encoding='utf8') as csvfile1:
 #   spamwriter = csv.writer(csvfile1, delimiter=',',
  #                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
   # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])