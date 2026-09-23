import gzip, os, numpy
names = ["train-images-idx3-ubyte.gz" ,
"train-labels-idx1-ubyte.gz",
"t10k-images-idx3-ubyte.gz",
"t10k-labels-idx1-ubyte.gz" ]

def get_data(i):
    with gzip.open(os.path.join("data", "mnist", i), 'rb') as f:
        data = f.read()
    mnumber = int.from_bytes(data[0:4])
    if mnumber == 2051:
        cpictures = int.from_bytes(data[4:8])
        rows = int.from_bytes(data[8:12])
        cols = int.from_bytes(data[12:16])
        arr = numpy.frombuffer(data, dtype=numpy.uint8,offset=16)
        arr = arr.reshape(cpictures, rows*cols)
        arr = arr.astype(numpy.float32)/255.0
    else:
        arr = arr = numpy.frombuffer(data, dtype=numpy.uint8, offset=8)
        arr = arr.astype(numpy.int64)
    return arr
x_train = get_data(names[0])
x_test = get_data(names[1])
y_train = get_data(names[2])
y_test = get_data(names[3])
if __name__ == "__main__":
    print(x_train.shape, x_train.max(), x_train.dtype)