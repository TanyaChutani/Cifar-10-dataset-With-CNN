import numpy as np
import seaborn as sn
import pandas as pd
from matplotlib import pyplot
from keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.optimizers import SGD

np.random.seed(14)

from scipy.misc import toimage

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))
pyplot.show()
#4Layer
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train= (X_train/255)
X_test= (X_test/255)
Y_train =to_categorical(Y_train, 10)
Y_test =to_categorical(Y_test, 10)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.summary()
cnn=model.fit(X_train, Y_train, batch_size= 100,
              shuffle=True, epochs=20, verbose= 1)
l, a= model.evaluate(X_test, Y_test)
print("Loss: ", l)
print("Accuracy: ", a)
#72

Y_pred = model.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(Y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(Y_test,axis=1),y_pred)
print(cm)
 
df_cm = pd.DataFrame(cm, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
'''
Epoch 1/20
50000/50000 [==============================] - 155s 3ms/step - loss: 1.7221 - acc: 0.3662
Epoch 2/20
50000/50000 [==============================] - 147s 3ms/step - loss: 1.3935 - acc: 0.4996
Epoch 3/20
50000/50000 [==============================] - 146s 3ms/step - loss: 1.2447 - acc: 0.5570
Epoch 4/20
50000/50000 [==============================] - 147s 3ms/step - loss: 1.1540 - acc: 0.5921
Epoch 5/20
50000/50000 [==============================] - 146s 3ms/step - loss: 1.0899 - acc: 0.6134
Epoch 6/20
50000/50000 [==============================] - 146s 3ms/step - loss: 1.0305 - acc: 0.6368
Epoch 7/20
50000/50000 [==============================] - 146s 3ms/step - loss: 0.9840 - acc: 0.6522
Epoch 8/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.9478 - acc: 0.6657
Epoch 9/20
50000/50000 [==============================] - 148s 3ms/step - loss: 0.9127 - acc: 0.6803
Epoch 10/20
50000/50000 [==============================] - 146s 3ms/step - loss: 0.8813 - acc: 0.6926
Epoch 11/20
50000/50000 [==============================] - 149s 3ms/step - loss: 0.8551 - acc: 0.6990
Epoch 12/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.8314 - acc: 0.7079
Epoch 13/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.8097 - acc: 0.7145
Epoch 14/20
50000/50000 [==============================] - 146s 3ms/step - loss: 0.7947 - acc: 0.7193
Epoch 15/20
50000/50000 [==============================] - 146s 3ms/step - loss: 0.7696 - acc: 0.7286
Epoch 16/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.7486 - acc: 0.7362
Epoch 17/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.7347 - acc: 0.7394
Epoch 18/20
50000/50000 [==============================] - 148s 3ms/step - loss: 0.7187 - acc: 0.7456
Epoch 19/20
50000/50000 [==============================] - 147s 3ms/step - loss: 0.7075 - acc: 0.7477
Epoch 20/20
50000/50000 [==============================] - 150s 3ms/step - loss: 0.6998 - acc: 0.7510
10000/10000 [==============================] - 15s 1ms/step
Loss:  0.805845506001
Accuracy:  0.7238
'''


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train= (X_train/255)
X_test= (X_test/255)
Y_train =to_categorical(Y_train, 10)
Y_test =to_categorical(Y_test, 10)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
cnn=model.fit(X_train, Y_train, batch_size= 100,
              shuffle=True, epochs=20, verbose= 1)

l, a= model.evaluate(X_test, Y_test)
print("Loss: ", l)
print("Accuracy: ", a)
#72

Y_pred = model.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(Y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(Y_test,axis=1),y_pred)
print(cm)
 
df_cm = pd.DataFrame(cm, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
'''
_________________________________________________________________
Epoch 1/20
50000/50000 [==============================] - 459s 9ms/step - loss: 1.8322 - acc: 0.3101
Epoch 2/20
50000/50000 [==============================] - 330s 7ms/step - loss: 1.4937 - acc: 0.4585
Epoch 3/20
50000/50000 [==============================] - 307s 6ms/step - loss: 1.3524 - acc: 0.5163
Epoch 4/20
50000/50000 [==============================] - 294s 6ms/step - loss: 1.2490 - acc: 0.5551
Epoch 5/20
50000/50000 [==============================] - 298s 6ms/step - loss: 1.1722 - acc: 0.5852
Epoch 6/20
50000/50000 [==============================] - 297s 6ms/step - loss: 1.1092 - acc: 0.6081
Epoch 7/20
50000/50000 [==============================] - 299s 6ms/step - loss: 1.0516 - acc: 0.6313
Epoch 8/20
50000/50000 [==============================] - 297s 6ms/step - loss: 0.9961 - acc: 0.6475
Epoch 9/20
50000/50000 [==============================] - 308s 6ms/step - loss: 0.9552 - acc: 0.6635
Epoch 10/20
50000/50000 [==============================] - 326s 7ms/step - loss: 0.9132 - acc: 0.6801
Epoch 11/20
50000/50000 [==============================] - 299s 6ms/step - loss: 0.8747 - acc: 0.6933
Epoch 12/20
50000/50000 [==============================] - 298s 6ms/step - loss: 0.8368 - acc: 0.7092
Epoch 13/20
50000/50000 [==============================] - 302s 6ms/step - loss: 0.8076 - acc: 0.7173
Epoch 14/20
50000/50000 [==============================] - 296s 6ms/step - loss: 0.7782 - acc: 0.7283
Epoch 15/20
50000/50000 [==============================] - 306s 6ms/step - loss: 0.7575 - acc: 0.7351
Epoch 16/20
50000/50000 [==============================] - 303s 6ms/step - loss: 0.7294 - acc: 0.7445
Epoch 17/20
50000/50000 [==============================] - 310s 6ms/step - loss: 0.7007 - acc: 0.7544
Epoch 18/20
50000/50000 [==============================] - 305s 6ms/step - loss: 0.6769 - acc: 0.7632
Epoch 19/20
50000/50000 [==============================] - 301s 6ms/step - loss: 0.6594 - acc: 0.7684
Epoch 20/20
50000/50000 [==============================] - 298s 6ms/step - loss: 0.6355 - acc: 0.7763
10000/10000 [==============================] - 44s 4ms/step
Loss:  0.850518771362
Accuracy:  0.7244

'''
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train= (X_train/255)
X_test= (X_test/255)
Y_train =to_categorical(Y_train, 10)
Y_test =to_categorical(Y_test, 10)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
cnn=model.fit(X_train, Y_train, batch_size= 100,
              shuffle=True, epochs=20, verbose= 1)

l, a= model.evaluate(X_test, Y_test)
print("Loss: ", l)
print("Accuracy: ", a)
#74
Y_pred = model.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(Y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(Y_test,axis=1),y_pred)
print(cm)
 
df_cm = pd.DataFrame(cm, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
'''
______________________________________________________________
Epoch 1/20
50000/50000 [==============================] - 383s 8ms/step - loss: 1.7113 - acc: 0.3697
Epoch 2/20
50000/50000 [==============================] - 369s 7ms/step - loss: 1.3592 - acc: 0.5127
Epoch 3/20
50000/50000 [==============================] - 368s 7ms/step - loss: 1.1935 - acc: 0.5764
Epoch 4/20
50000/50000 [==============================] - 367s 7ms/step - loss: 1.0741 - acc: 0.6238
Epoch 5/20
50000/50000 [==============================] - 367s 7ms/step - loss: 0.9850 - acc: 0.6567
Epoch 6/20
50000/50000 [==============================] - 406s 8ms/step - loss: 0.9242 - acc: 0.6776
Epoch 7/20
50000/50000 [==============================] - 366s 7ms/step - loss: 0.8650 - acc: 0.6987
Epoch 8/20
50000/50000 [==============================] - 364s 7ms/step - loss: 0.8309 - acc: 0.7099
Epoch 9/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.7933 - acc: 0.7233
Epoch 10/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.7590 - acc: 0.7338
Epoch 11/20
50000/50000 [==============================] - 365s 7ms/step - loss: 0.7279 - acc: 0.7447
Epoch 12/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.7044 - acc: 0.7528
Epoch 13/20
50000/50000 [==============================] - 376s 8ms/step - loss: 0.6712 - acc: 0.7619
Epoch 14/20
50000/50000 [==============================] - 416s 8ms/step - loss: 0.6515 - acc: 0.7678
Epoch 15/20
50000/50000 [==============================] - 411s 8ms/step - loss: 0.6309 - acc: 0.7779
Epoch 16/20
50000/50000 [==============================] - 364s 7ms/step - loss: 0.6103 - acc: 0.7813
Epoch 17/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.5953 - acc: 0.7863
Epoch 18/20
50000/50000 [==============================] - 364s 7ms/step - loss: 0.5688 - acc: 0.7932
Epoch 19/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.5531 - acc: 0.8004
Epoch 20/20
50000/50000 [==============================] - 363s 7ms/step - loss: 0.5413 - acc: 0.8035
10000/10000 [==============================] - 37s 4ms/step
Loss:  0.819005190086
Accuracy:  0.7481
'''
#6 Layer
seed = 7
import numpy
numpy.random.seed(seed)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
Y_pred = model.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)
 
df_cm = pd.DataFrame(cm, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()

'''
Epoch 1/25
50000/50000 [==============================] - 1151s 23ms/step - loss: 1.9426 - acc: 0.2844 - val_loss: 1.6577 - val_acc: 0.4034
Epoch 2/25
50000/50000 [==============================] - 1152s 23ms/step - loss: 1.5199 - acc: 0.4467 - val_loss: 1.3587 - val_acc: 0.5114
Epoch 3/25
50000/50000 [==============================] - 1162s 23ms/step - loss: 1.3531 - acc: 0.5087 - val_loss: 1.2909 - val_acc: 0.5339
Epoch 4/25
50000/50000 [==============================] - 1087s 22ms/step - loss: 1.2304 - acc: 0.5578 - val_loss: 1.1801 - val_acc: 0.5710
Epoch 5/25
50000/50000 [==============================] - 1228s 25ms/step - loss: 1.1230 - acc: 0.5984 - val_loss: 1.0956 - val_acc: 0.6076
Epoch 6/25
50000/50000 [==============================] - 1192s 24ms/step - loss: 1.0336 - acc: 0.6316 - val_loss: 0.9866 - val_acc: 0.6561
Epoch 7/25
50000/50000 [==============================] - 1093s 22ms/step - loss: 0.9574 - acc: 0.6602 - val_loss: 0.8881 - val_acc: 0.6839
Epoch 8/25
50000/50000 [==============================] - 1098s 22ms/step - loss: 0.8913 - acc: 0.6834 - val_loss: 0.8457 - val_acc: 0.7009
Epoch 9/25
50000/50000 [==============================] - 1004s 20ms/step - loss: 0.8348 - acc: 0.7034 - val_loss: 0.8463 - val_acc: 0.6993
Epoch 10/25
50000/50000 [==============================] - 997s 20ms/step - loss: 0.7836 - acc: 0.7220 - val_loss: 0.7822 - val_acc: 0.7226
Epoch 11/25
50000/50000 [==============================] - 994s 20ms/step - loss: 0.7447 - acc: 0.7356 - val_loss: 0.7531 - val_acc: 0.7365
Epoch 12/25
50000/50000 [==============================] - 995s 20ms/step - loss: 0.7041 - acc: 0.7496 - val_loss: 0.7381 - val_acc: 0.7409
Epoch 13/25
50000/50000 [==============================] - 1009s 20ms/step - loss: 0.6762 - acc: 0.7595 - val_loss: 0.7038 - val_acc: 0.7523
Epoch 15/25
50000/50000 [==============================] - 1306s 26ms/step - loss: 0.6231 - acc: 0.7805 - val_loss: 0.6930 - val_acc: 0.7570
Epoch 16/25
50000/50000 [==============================] - 1280s 26ms/step - loss: 0.5953 - acc: 0.7897 - val_loss: 0.6831 - val_acc: 0.7623
Epoch 17/25
50000/50000 [==============================] - 993s 20ms/step - loss: 0.5705 - acc: 0.7985 - val_loss: 0.6757 - val_acc: 0.7648
Epoch 18/25
50000/50000 [==============================] - 992s 20ms/step - loss: 0.5515 - acc: 0.8046 - val_loss: 0.6640 - val_acc: 0.7713
Epoch 19/25
50000/50000 [==============================] - 991s 20ms/step - loss: 0.5332 - acc: 0.8106 - val_loss: 0.6566 - val_acc: 0.7703
Epoch 20/25
50000/50000 [==============================] - 993s 20ms/step - loss: 0.5237 - acc: 0.8120 - val_loss: 0.6395 - val_acc: 0.7777
Epoch 21/25
50000/50000 [==============================] - 991s 20ms/step - loss: 0.5002 - acc: 0.8216 - val_loss: 0.6391 - val_acc: 0.7811
Epoch 22/25
50000/50000 [==============================] - 991s 20ms/step - loss: 0.4907 - acc: 0.8229 - val_loss: 0.6426 - val_acc: 0.7803
Epoch 23/25
50000/50000 [==============================] - 992s 20ms/step - loss: 0.4693 - acc: 0.8329 - val_loss: 0.6324 - val_acc: 0.7840
Epoch 24/25
50000/50000 [==============================] - 991s 20ms/step - loss: 0.4556 - acc: 0.8369 - val_loss: 0.6302 - val_acc: 0.7859
Epoch 25/25
50000/50000 [==============================] - 991s 20ms/step - loss: 0.4417 - acc: 0.8423 - val_loss: 0.6343 - val_acc: 0.7823
Accuracy: 78.23%
0 1000
1 1000
2 1000
3 1000
4 1000
5 1000
6 1000
7 1000
8 1000
9 1000
[[818  14  40   4  13   0   7   6  74  24]
 [ 11 902   5   1   1   1   4   1  23  51]
 [ 56   3 723  48  59  24  51  20  11   5]
 [ 24   9  69 614  59 113  57  20  20  15]
 [ 16   2  71  42 767  16  31  43   9   3]
 [ 13   3  60 206  53 593  21  39   8   4]
 [ 10   3  38  43  25  10 853   2  10   6]
 [ 11   1  47  32  76  21   4 798   6   4]
 [ 36  22   9  11   2   4   3   0 904   9]
 [ 32  56   9   7   1   2   4   8  30 851]]
'''