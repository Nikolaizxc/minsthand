import numpy as np
import data_loader as dl

def init_params(n):
    #n - chislo neuronov
    w1 = np.random.randn(784, n) * np.sqrt(2.0 / 784)
    b1 = np.zeros(n)

    w2 = np.random.randn(n, n)*np.sqrt(2/n)* np.sqrt(2.0 / 16)
    b2 = np.zeros(n)
    w3 = np.random.randn(n,10)* np.sqrt(2.0 / 16)
    b3 = np.zeros(10)
    return [w1,b1,w2,b2,w3,b3]
def forward(X,params):
    #х - массив точек парамс - веса(в1 б1 в2 б2, w3, b3)
    Z1 = X@params[0] + params[1]
    A1 = np.maximum(0, Z1)
    Z2 = A1@params[2] + params[3]
    A2 = np.maximum(0,Z2)
    Z3 = A2@params[4] + params[5]
    #print([i.shape for i in [X, Z1, A1, Z2, A2, Z3]])
    return [X, Z1, A1, Z2, A2, Z3]
def softmax(Z2):
    #з2 - итоги форвард
    res = np.array(Z2)
    #print(res)
    res -= np.max(res, axis=1, keepdims=True)
    res = np.exp(res)
    res  = res/np.sum(res,axis=1,keepdims=1)
    #print(res)


    return res
def backward(res, y, params,start, lr):
  
    #рес - итоговый ответ з2, у - правильные ответы, парамс - конечнная обработка [X, Z1, A1, Z2, A2, Z3],старт - начальные веса[w1,b1,w2,b2,w3,b3] , lr eto shag s kotorym idet neiroset
    targets = np.array([[0 if n != k else 1 for n in range(10)] for k in y])
    #print(targets.shape, res.shape, res[0],  targets[0])
    Res = res-targets

    delta1 = (Res @ start[4].T) * (params[4] > 0)
    delta2 = (delta1 @ start[2].T)*(params[2] > 0)
    #print(start[3], start[1])
    
    start[5] -= np.sum(Res, axis=0) *lr / len(Res)
    start[4] -= (params[4].T @ Res)*lr / len(Res)
    
    start[3] -= np.sum(delta1, axis=0) *lr / len(Res)
    start[2] -= (params[2].T @ delta1)*lr / len(Res)
   
    start[1] -= np.sum(delta2, axis=0)*lr/len(Res)
    start[0] -= (params[0].T @ delta2) *lr/len(Res)
    return start
def cross(res, ans):
    eps = 1e-12
    # Выбираем вероятности для истинных классов с помощью индексации NumPy
    prob_correct = res[np.arange(len(ans)), ans]
    return -np.mean(np.log(prob_correct + eps))


def right(res, ans):
    # Находим индекс максимального элемента в каждой строке и сравниваем с ответом
    return np.sum(np.argmax(res, axis=1) == ans)

def getweights(data):
    np.savez("weights_for_MNIST", *data)
    return 
if __name__ == "__main__":
    b = init_params(128)

    for epoch in range(10):
        cnt = 0
        sred = 0
        true = 0
        while True:
            mass_imgs = np.array(dl.x_train[cnt:min(cnt+64, 60000)])
            mass_ans = np.array(dl.x_test[cnt:min(cnt+64, 60000)])
            forw = forward(mass_imgs, b)
            end = softmax(forw[5])
            b = backward(end, mass_ans, forw, b, 0.1)
            sred+= cross(end, mass_ans)
            true += right(end, mass_ans)
            cnt+=64
            if cnt%(64*20) == 0:
                print(cnt)
            if cnt >= 60000:
                break

        print("loss" + str(sred/(60000/64)), "current epoch:"+str(epoch+1), "right answers from 60000:"+str(true))


    getweights(b)