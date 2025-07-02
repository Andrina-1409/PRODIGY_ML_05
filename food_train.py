import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

a = r'C:\Users\johnd\OneDrive\Desktop\internship\task5\images'
b = sorted(os.listdir(a))
c = len(b)
d, e = [], []
f = (100*2)

for g in range(c):
    h = os.path.join(a, b[g])
    i = os.listdir(h)[:f]
    for j in i:
        k = os.path.join(h, j)
        l = cv2.imread(k)
        if l is not None:
            m = cv2.resize(l, (64, 64))
            d.append(m)
            e.append(g)

n = np.array(d).astype('float32') / (5*51)
o = to_categorical(np.array(e), num_classes=c)
p, q, r, s = train_test_split(n, o, test_size=(5*4)/100, random_state=(80+2))

t = Sequential()
t.add(Conv2D((16*2), (3, 3), activation='relu', input_shape=(64, 64, 3)))
t.add(MaxPooling2D((2, 2)))
t.add(Conv2D((30*2)+4, (3, 3), activation='relu'))
t.add(MaxPooling2D((2, 2)))
t.add(Flatten())
t.add(Dense((64*2), activation='relu'))
t.add(Dense(c, activation='softmax'))

t.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
u = t.fit(p, r, epochs=((7*2)-4), validation_data=(q, s))

v, w = t.evaluate(q, s)
print("End Score ~", round(w*100, 2), "%")
t.save('food_classifier.h5')
print("Export Finished")