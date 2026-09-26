import numpy as np
import matplotlib.pyplot as plt


def village(seed):
    seed = np.random.default_rng(seed)
    points = np.array(seed.uniform(-0.3, 0.3, 2000))
    
    ans = list()
    points = points.reshape(1000, 2)
    a = seed.integers(0,2, 1000)
    cnt = 0
    for id, i in enumerate(points):
        

        x,y = i[0], i[1]
        if a[cnt]:
            points[id][0] += 0.5
            points[id][1] += 0.5
            ans.append(1)
        else:
            points[id][0] -= 0.5
            points[id][1] -= 0.5
            ans.append(0)
        cnt+=1
    return np.column_stack((points, ans))

def xor(seed):
    seed = np.random.default_rng(seed)
    points = np.array(seed.uniform(-1, 1, 2000))
    points = points.reshape(1000,2)
    ans = list()
    for i in points:
        
        ans.append(int((i[0]*i[1])>0))
    return np.column_stack((points, ans))
def ring(seed):
    seed = np.random.default_rng(seed)
    points = np.array(seed.uniform(0, 1, 2000))
    points = points.reshape(1000,2)
    ans = list()

    for id, i in enumerate(points):
        a = i[0]
        if abs(i[0]) > 0.5:
            ans.append(1)
            
            points[id][0] = ((a * np.sin(i[1]*10000)))
            points[id][1] = (a* np.cos(i[1]*10000))
        else:
            ans.append(0)
            points[id][0] = ((a * np.sin(i[1]*10000)))
            points[id][1] = ((a * np.cos(i[1]*10000)))

    return np.column_stack((points, ans))
def spray(seed):
    seed2 = np.random.default_rng(seed+1)
    seed = np.random.default_rng(seed)
    points = np.array(seed.uniform(0, 1, 2000))
    points = points.reshape(1000,2)
    cl = seed.integers(0,2, 1000)
    vx = seed.uniform(0,0.25,1000)
    vy = (seed2).uniform(0,0.25,1000)
    ans = list()
    for id, i in enumerate(points):
        a = i[0]
        b = i[1]
        k = 10
        dist = (a**2 + b**2)**0.5
        if (cl[id]):
            points[id][0] = (dist*k)*np.cos(dist*4*np.pi) +vx[id]
            points[id][1] = (dist*k)*np.sin(dist*4*np.pi)+vy[id]
            
        else:
            points[id][0] = (dist*k)*np.cos(np.pi +dist*4*np.pi)+vx[id]
            points[id][1] = (dist*k)*np.sin(np.pi +dist*4*np.pi)+vy[id]
        ans.append(cl[id])
    
    points/=15

        

    return np.column_stack((points, ans))

if __name__ == "__main__":
    x,y,x1,y1 = [],[],[],[]
    points = spray(1)
    for i in points:
        if i[2]:
            x.append(i[0])
            y.append(i[1])
        else:
            x1.append(i[0])
        
            y1.append(i[1])
    plt.scatter(x1, y1, color = 'blue', s = 15)
    plt.scatter(x, y, color = 'red', s = 15)
    plt.show()