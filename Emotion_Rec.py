import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization

train=pd.read_csv('train.csv')
t=train.pixels.tolist()

x=[]
for i in t:
    a=[int(x1) for x1 in i.split()]
    a=np.asarray(a).reshape(48,48)
    x.append(a.astype('float32'))

X=np.asarray(x)
X=np.expand_dims(X,-1)
y = pd.get_dummies(train['emotion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


######## MODEL CREATION

model=Sequential()

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(7,activation='sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=32,verbose=1,epochs=5,validation_data=(X_test,y_test))

#### PREDICTION OF AN IMAGE

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

img = cv2.imread("test.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.3  , 10)

for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        fi_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        out= model.predict(fi_img)
        cv2.putText(img, labels[int(np.argmax(out))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(out))])

cv2.imshow('Emotion', img)
cv2.waitKey()
