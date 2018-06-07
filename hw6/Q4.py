from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from train import *
_mean, _std = mean_std()

def get_label(movie='movies.csv'):
    r = open(movie, encoding='latin-1').read()
    l = r.split('\n')[1:-1]
    m_data = []
    index = []
    yr = []
    for i in range(len(l)):
        line = l[i].split('::')
        index.append(int(line[0]))
        m_data.append(line[2].split('|'))
        yr.append(int(line[1][-5:-1]))

    yr = np.array(yr, dtype=float)
    yr = (yr - yr.mean()) / yr.std()


    plane = [j for i in m_data for j in i]
    category = []
    for i in plane:
        if i not in category:
                category.append(i)
    index = np.array(index, dtype=int)
    movie_mat = np.zeros((index.max(),len(category)+1))
    for i in range(len(index)):
        one = [category.index(j) for j in m_data[i]]
        movie_mat[index[i]-1, one] = 1
    for i in range(len(movie_mat)):
        if movie_mat[i].sum():
            movie_mat[i]/=movie_mat[i].sum()
    x = [[],[],[]]
    y = []
    weight = get_emb()
    for i in range(len(index)):
        a = np.zeros((3,))
        if movie_mat[index[i]-1,0:2].sum()!=0:
            a[0] = 1
        if movie_mat[index[i]-1,7:9].sum()!=0:
            a[1] = 1
        if movie_mat[index[i]-1,12:14].sum()!=0:
            a[2] = 1
        if a.sum() == 1:
            x[np.argmax(a)].append(weight[index[i]-1])

    return np.array(x[0]+x[1]+x[2]), [len(x[0]), len(x[1]), len(x[2])]


def get_emb(model_name='model_84794.h5'):
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    #model.compile(loss='mse', optimizer=sgd, metrics=[rmse])
    model = load_model(model_name, custom_objects={'rmse':rmse})
    movie_emb = np.array(model.layers[5].get_weights()).squeeze()
    return movie_emb

def draw(x, y):
    vis = TSNE(n_components=2).fit_transform(x)
    sc = plt.scatter(vis[:y[0], 0], vis[:y[0], 1], color='red')
    sc = plt.scatter(vis[y[0]:y[0]+y[1], 0], vis[y[0]:y[0]+y[1], 1], color='green')
    sc = plt.scatter(vis[y[0]+y[1]:, 0], vis[y[0]+y[1]:, 1], color='blue')
    plt.show()
    return vis

if __name__ == '__main__':
    x, y = get_label()
    res = draw(x, y)