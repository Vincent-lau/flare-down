with open ("fd-export.csv","r") as f1:
    with open("txtFileOfCsv.txt","w") as f2:
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