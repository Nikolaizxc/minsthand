import tkinter as tk
import knn
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
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

root.mainloop()
