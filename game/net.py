import numpy as np
import matplotlib.pyplot as plt

def init_params(n):
    #н - число нейронов 
    w1 = np.random.randn(2, n)
    b1 = np.random.uniform(-1,1,n)
    w2 = np.random.randn(n, n)*np.sqrt(2/n)
    b2 = np.random.uniform(-1,1,n)
    w3 = np.random.randn(n,2)*0.1
    b3 = np.random.uniform(-1,1,2)
    return [w1,b1,w2,b2, w3, b3]


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
    res -= np.max(res, axis=1, keepdims=True)
    res = np.exp(res)
    res  = res/np.sum(res,axis=1,keepdims=1)


    return res

def backward(res, y, params,start, lr):
  
    #рес - итоговый ответ з2, у - правильные ответы, парамс - конечнная обработка [X, Z1, A2, Z2, A3, Z3],старт - начальные веса[w1,b1,w2,b2,w3,b3] , lr eto shag s kotorym idet neiroset
    targets = np.array([[1,0] if k == 0 else [0,1] for k in y])
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
def loss(res, anss):
    result = 0

    for i in range(len(res)):
        if max(res[i]) == res[i][anss[i]]:
            
            result+=1
    return result
def cross(res, anss):
    #рес - результат работы, анс - правильные ответы

    s = 0
    for id, i in enumerate(res):
        
        a = i[anss[id]] 

        s+=-np.log(a + 1e-12)
       
    return s/len(anss)
def getweights(data):
    np.savez("weights", *data)
    return 
def result(data):
    dots = []
    for x in range(-100, 100):
        for y in range(-100,100):
            dots.append([x/100, y/100])
    dots = np.array(dots)
    z = forward(dots, data)
    z = softmax(z[-1])
    plt.scatter(dots[:, 0], dots[:, 1], c=np.argmax(z, axis=1), cmap='bwr_r', vmin=0, vmax=1, s=15)
    plt.show()
    return
if __name__ == "__main__":
    import generator as G
    vil = G.spray(1)
    #получение уровня
    b = init_params(15)
    cros, right = 1,1
    vil, vil1 = vil
    count = 0 
    #инициализация весов в первый раз (случайная)
    for i in range(5000):
        if count == 20:
            print(i, cros, right)
            break
        if right > len(vil1)*0.98:
            vil = G.spray(i)
            vil, vil1 = vil
            count +=1
        else:
            count=0

        forw = forward(vil, b)



        end = softmax(forw[5])

        #итоговые результаты в процентах
        right = loss(end, vil1)
        #подсчет сколько правильно а сколько нет
        cros = cross(end,vil1)
        #подсчет уверенности (чем больше уверенность тем больше разрыв между правильным результатом и неправильным)
        b = backward(end, vil1, forw,b, 0.3)
        
        #корректировка весов
        if i == 100 or i == 1000 or i == 4000:
            print(np.mean(forw[2] > 0), np.mean(forw[4] > 0))
            print(cros, right)
    getweights(b)
else:
    import game.generator as G