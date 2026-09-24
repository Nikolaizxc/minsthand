import data_loader as dl, numpy as np

from PIL import Image
count = 0
def search_knn(k):
    test_img = k
    res = list()
    distances = np.linalg.norm(dl.x_train - test_img, axis=1)
    res = np.argsort(distances)[:10]
    res = dl.y_train[res]
    cur = dict()
    for i in res:
        if i in cur:

            cur[i]+=1
            continue
        cur[i] = 1
    z = (None, 0)

    for i in [x for x in cur.items()]:
        if z[1] < i[1]:
            z = i
    

    return z
if __name__ == "__main__":

    img = Image.open("example.png")
    img = np.array(img)
    img = img.astype(np.float32)/255.0
    img = img.reshape(784, )
    a = np.ones(784, dtype=np.float32)
    a = a.astype(np.float32)
    #img = a - img

    print(img.mean())
        