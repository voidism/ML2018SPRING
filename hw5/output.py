import csv

def output(ans,name='ans.csv', header=['id','label']):
    text = open(name, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(header)
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()