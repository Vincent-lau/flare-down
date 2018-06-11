import codecs

with codecs.open("/Users/liuliu/Desktop/Python/flare_down/fd-export2.csv", 'r',encoding="utf-8") as f1:
    with codecs.open("/Users/liuliu/Desktop/Python/flare_down/exVersion/txtFileOfCsv.txt","w",encoding="utf-8") as f2:
        for line in f1:
            line=line.rstrip('\n')
            if (line[-1] != ','):
                line+=','
            line+='\n'
            f2.write(line)



'''with open ("txtFileOfCsv.txt",'r') as f:
    for line in f:
        for s in line.split(','):
            if()'''