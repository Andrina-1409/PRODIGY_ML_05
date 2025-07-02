from tensorflow.keras.models import Model
import os
import numpy
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout ,GlobalAveragePooling2D,Dense

path=r'D:\Python projects\foodcalorieCNN\images'
cat=[]
lst=os.listdir(path)
for xx in lst:
 loc=path+'\\'+xx
 if os.path.isdir(loc):
  cat.append(xx)
cat.sort()
cl=len(cat)
img,lbl=[],[]
for ind,val in enumerate(cat):
 pth=os.path.join(path,val)
 temp=os.listdir(pth)[:150]  
 for ele in temp:
  full=os.path.join(pth,ele)
  im=cv2.imread(full)
  if im is not None:
   im=cv2.resize(im,(123,123))
   img.append(im)
   lbl.append(ind)
arr=np.array(img).astype('float32')/255
lab=to_categorical(np.array(lbl),num_classes=cl)
a1,a2,a3,a4=train_test_split(arr,lab,test_size=0.23,random_state=7)
dat=ImageDataGenerator(rotation_range=17,zoom_range=0.13,width_shift_range=0.07,height_shift_range=0.06,horizontal_flip=True)
dat.fit(a1)
base=MobileNetV2(weights='imagenet',include_top=False,input_shape=(123,123,3))
for lay in base.layers: lay.trainable=False
out=base.output
out=GlobalAveragePooling2D()(out)
out=Dense(173,activation='relu')(out)
out=Dropout(0.29)(out)
out=Dense(cl,activation='softmax')(out)
mod=Model(inputs=base.input,outputs=out)
mod.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
h=mod.fit(dat.flow(a1,a3,batch_size=29),epochs=19,validation_data=(a2,a4))
l,s=mod.evaluate(a2,a4)
print('Acc:',round(s*100,2))
mod.save('food_classifier.h5')
print('Finished')
