from skimage import io
import numpy as np
import os, sys

def load_data(path='./Aberdeen/*'):
    imgs = io.imread_collection(path)
    imgs_array = np.array(imgs)
    imgs_array = np.reshape(imgs_array, (415,-1))
    X = imgs_array.T.astype('float64')
    mean_face = np.mean(X, axis=1)
    X -= mean_face.reshape(1080000, 1)
    # np.save('X.npy', X)
    # np.save('mean_face.npy',mean_face)
    return X, mean_face

def get_SVD(X):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    # egs = np.reshape(U.copy(), (-1, 600, 600, 3))
    # np.save('./U.npy', U)
    # np.save('./s.npy', s)
    # np.save('./V.npy', V)
    return U, s, V
    
# for i in range(10):
#    io.imsave("eg"+str(i)+".jpg", np.reshape((U.T[i]+mean_face), (600, 600, 3)).astype('uint8'))
# X, m = load_data()
X = np.load('./X.npy')
m = np.load('./mean_face.npy')
U = np.load('./U.npy')
s = np.load('./s.npy')
V = np.load('./V.npy')

def eigenface(U):
    for i in range(10):
        M = -np.reshape((U.T[i]), (600, 600, 3))
        M -= np.min(M)
        M /= np.max(M)
        M = (M*255).astype('uint8')
        io.imsave("./faces/eg_" + str(i) + ".jpg", M)

def see_weight(U, s, V, X):
    S = np.diag(s)
    C = U.T.dot(X)
    W = S.dot(V)
    return C, W

def reconstruct(x, U, mean_face, dim=4, name='reconstruction.jpg'):
    mean_face = np.reshape(mean_face,(600, 600, 3))
    u = U[:,:dim]
    w = u.T.dot(x)
    res = u.dot(w)
    M = np.reshape(res, (600, 600, 3))
    M += mean_face
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype('uint8')
    io.imsave(name, M)

def ratio(idx,s , all=False):
    if all:
        return s[idx] / s.sum()
    else:
        return s[idx] / s[:4].sum()

def recons_from_img(path, U, mean_face, dim=4, name='reconstruction.jpg'):
    img = io.imread(path)
    img_array = np.array(img)
    img_array = img_array.reshape((600*600*3,))
    x = img_array.T.astype('float64')
    reconstruct(x, U, mean_face, dim, name)

if __name__ == '__main__':
    img_path = os.path.join(sys.argv[1], '*')
    X, m=load_data(img_path)
    U, s, V=get_SVD(X)
    recons_from_img(sys.argv[2], U, m)
    