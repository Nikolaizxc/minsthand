import tkinter as tk
import knn
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import percepthrone as pt
data = np.load("weights_for_MNIST.npz")
b = [data[k] for k in data.files]     #  список из шести массивов для передачи в форвард(готовые веса)

import sys
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # Для Windows 8.1 и 10/11
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware() # Для более старых Windows
        except:
            pass
root = tk.Tk()

root.title = "распознавание цифр"
root.geometry("400x600")
history = []
global r
r = 5
def percepthrone(event):
    root.update()
    x = root.winfo_rootx()+canvas.winfo_x()+2
    y = root.winfo_rooty()+canvas.winfo_y()+2
    x1 = x + 280 - 2
    y1 = y + 280 - 2
    
    img = ImageGrab.grab(bbox=(x, y, x1, y1))
    img = img.resize((28,28), Image.Resampling.BOX)
    img = img.convert("L")
    img = np.array(img)

    img = img.astype(np.float32)/255.0
    img = img.reshape(28,28)
    cnt = 0
    sum_x = 0
    sum_y = 0
    S, N, W, E = 0,-1,-1, 0
    for i in img:
    

        if max(i) > 0.1:
            #print(max(i), cnt, N, S)
            if N==-1:

                N = cnt
            S = cnt
        cnt+=1

    img = img.T
    cnt = 0
    for i in img:
        
        if max(i) > 0.1:
            if W == -1:
                W = cnt
            E = cnt
        cnt+=1
    s = []
    
    #plt.imshow(img, cmap='grey')
    #plt.show()

    for cnt, i in enumerate(img.T):
        if N <= cnt <= S:
            s.append(i[W:E+1])
    s = np.array(s)
    #plt.imshow(s, cmap='grey')
    #plt.show()
    x_index = np.arange(s.shape[1])
    y_index= np.arange(s.shape[0])
    sum_x = np.sum(s.sum(axis=0) * x_index) / np.sum(s)
    sum_y = np.sum(s.sum(axis=1) * y_index) / np.sum(s)
    sum_x = int(sum_x) if abs(int(sum_x) - sum_x) < 0.5 else int(sum_x) + 1
    sum_y = int(sum_y) if abs(int(sum_y) - sum_y) < 0.5 else int(sum_y) + 1
    ss = np.zeros((28, s.shape[1]))
    cnt = 0
    z_x = 0
    for i in range( 14 - sum_y, 14 - sum_y + s.shape[0]):
        
        ss[i] = s[cnt]
        z_x += sum([t*t_c for t_c, t in enumerate(ss[i])])
        cnt+=1
    print(z_x / np.sum(ss))
    sa = np.zeros((28, 28))
    cnt = 0

    for i in range( 14 - sum_x, 14 - sum_x + s.shape[1]):
        
        sa[i] = ss.T[cnt]

        # plt.imshow(sa, cmap='gray')
        # plt.show()
        cnt+=1

    img = sa.T
    img = img.reshape(1,784)
    ans = pt.softmax(pt.forward(img, b)[5]) [0]    
    print([str(z) + " " + str(i) for z, i in enumerate(ans)] )
    label_accuracy1["text"] = "Цифра: " + str(np.argmax(ans))
    label_accuracy["text"] = "Точность: "+f"{max(ans):.2f}" + "%"
def kn(event):
    root.update()
    x = root.winfo_rootx()+canvas.winfo_x()
    y = root.winfo_rooty()+canvas.winfo_y()
    x1 = x + 280
    y1 = y + 280
    
    img = ImageGrab.grab(bbox=(x, y, x1, y1))
    img = img.resize((28,28), Image.Resampling.BOX)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255.0
    img = img.reshape(784,)
    a = knn.search_knn(img)
    label_accuracy1["text"] = "Цифра: " + str(int(a[0]))
    label_accuracy["text"] = "Точность: "+str(a[1]*10) + "%"
    img = img.reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
def cls(event):
    while history:
        a = history.pop()
        if a!= 'END':
            canvas.delete(a)
def on_release(event):
    global r
    r = int(brush_size_scale.get()/2)
def f(event):
    global r
    x = event.x
    y = event.y
    
    id = canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
    history.append(id)
def br(event):
    if history:
        a = history.pop()
    while (history and history[-1] != "END"):
        a = history.pop()
        canvas.delete(a)
def end(event):
    history.append("END")
canvas = tk.Canvas(root, width=280, height=280, bg='black')
canvas.pack(pady=10)



brush_size_scale = tk.Scale(
    from_=10, 
    
    to=100, 
    
    orient=tk.HORIZONTAL
)


brush_size_scale.pack()









slidebar = tk.Scale(orient=tk.HORIZONTAL)
button_clear = tk.Button(root, text="Очистить", font=("Arial", 14))
button_clear.pack(pady=5)


label_accuracy1 = tk.Label(root, text="Цифра: -", font=("Arial", 14))
label_accuracy1.pack()
label_accuracy = tk.Label(root, text="Точность: -", font=("Arial", 14))
label_accuracy.pack()
button_recognize = tk.Button(root, text="Распознать knn", font=("Arial", 14))
button_recognize.pack(pady=5)

button_recognize2 = tk.Button(root, text="Распознать другим способом", font=("Arial", 14))
button_recognize2.pack(pady=5)

canvas.bind('<B1-Motion>', f)
root.bind('<Control-z>', br)
canvas.bind('<ButtonRelease-1>', end)
button_clear.bind('<Button-1>', cls)
brush_size_scale.bind("<ButtonRelease-1>", on_release)
button_recognize.bind('<Button-1>', kn)
button_recognize2.bind('<Button-1>', percepthrone)

root.mainloop()
