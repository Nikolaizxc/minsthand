import os, urllib.request
url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
names = ["train-images-idx3-ubyte.gz" ,
"train-labels-idx1-ubyte.gz",
"t10k-images-idx3-ubyte.gz",
"t10k-labels-idx1-ubyte.gz" ]
os.makedirs(os.path.join("data", "mnist"), exist_ok= True)


for i in names:
    path = os.path.join("data", "mnist", i)
    print("start")
    if os.path.exists(path) == False:
        urllib.request.urlretrieve(url+i, path)
    print("file download")
        

    
