from train import *
from sklearn.linear_model import LogisticRegression
import pickle
import sys

def skfit(dropnum=0):
    X, Y, Vx, Vy, T = load_data(v_size=1,rand=False,dropnum=dropnum,bias=0)
    classifier = LogisticRegression(penalty='l1')
    classifier.fit(X, Y)
    print(classifier.score(Vx,Vy))
    pickle.dump(classifier,open('sk_model.sav','wb'))
    classifier = pickle.load(open('sk_model.sav','rb'))
    Vans = classifier.predict(Vx)
    validSK(Vans, Vy)
    Xans = classifier.predict(X)
    validSK(Xans, Y)
    ans = classifier.predict(T)
    print(ans)


    text = open('skans.csv', "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow([int(i+1),int(ans[i])])
    text.close()


def validSK(ans,Vy):
    idx = 0
    sqrtsum = 0
    roundsum = 0
    succ = 0
    fail = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for res, j in zip(ans, Vy):
        if res>=0.5:
            res=1
        else:
            res = 0
        if res == j:
            succ+=1
        else:
            fail+=1
        if j==1:
            if res==1:
                TP+=1
            else:
                FN+=1
        else:
            if res==1:
                FP +=1
            else:
                TN+=1
        idx += 1
    print("score:", (succ / idx), "F1-measure:", (2*TP/(2*TP+FP+FN)))
    return (succ / idx)