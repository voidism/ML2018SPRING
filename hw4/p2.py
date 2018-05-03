import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
import csv
from output import output
import os, sys

def main(a, filename, output_name):
    #a = StandardScaler().fit_transform(a)
    pca = PCA(n_components=400, whiten=True)
    p = pca.fit_transform(a)
    for i in range(3):
        k = KMeans(n_clusters=2, random_state=100).fit(p)
        # np.save('klabels.npy', k.labels_)
        print('ksum:',k.labels_.sum())
        if k.labels_.sum()==70000:
            break
    r = csv.reader(open(filename))
    l = list(r)[1:]
    ans = []
    for i in l:
        i1 = int(i[1])
        i2 = int(i[2])
        if int(k.labels_[i1])==int(k.labels_[i2]):
            ans.append([i[0], '1'])
        else:
            ans.append([i[0], '0'])

    output(ans, name=output_name)

def predict(a, filename, output_name):
    k = np.load('klabels.npy')
    r = csv.reader(open(filename))
    l = list(r)[1:]
    ans = []
    for i in l:
        i1 = int(i[1])
        i2 = int(i[2])
        if int(k[i1])==int(k[i2]):
            ans.append([i[0], '1'])
        else:
            ans.append([i[0], '0'])

    output(ans, name=output_name)

if __name__ == '__main__':
    a = np.load(sys.argv[1])
    filename = sys.argv[2]
    predict(a, filename, output_name=sys.argv[3])
