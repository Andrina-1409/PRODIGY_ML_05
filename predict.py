import numpy as x
import cv2 as y
import os as z
from tensorflow.keras.models import load_model as l

a = l(r'D:\Python projects\foodcalorieCNN\food_classifier.h5')
b = r'D:\Python projects\foodcalorieCNN\images'
c = sorted(z.listdir(b))
d = x.linspace((5*11), (495-0), num=len(c))

def e(f):
    g = y.imread(f)
    if g is None:
        print("No Read:", f)
        return None
    h = y.resize(g, (64, 64))
    return (h/255).reshape(1, 64, 64, 3)

i = r'D:\Python projects\foodcalorieCNN\chicken.jpg'
j = e(i)

if j is not None:
    k = a.predict(j)
    l1 = x.argmax(k[0])
    m = c[l1]
    n = round(d[l1], 2)
    print(f"Item ~> {m}")
    print(f"Energy ~> {n} kcal")
