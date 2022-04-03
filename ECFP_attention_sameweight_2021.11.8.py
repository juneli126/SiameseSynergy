import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import locale
import os
from matplotlib.lines import Line2D
import io
from PIL import Image


def attention(x):
    pool_k = x.shape[2]
    dense_l = x.shape[3]
    x_real = x
    # print('x_real shape: ',x_real.shape)
    x = AveragePooling2D(pool_size=(pool_k), padding="same")(x)
    # print('x shape: ',x.shape)
    x = Dense(dense_l / 2)(x)
    # print('x shape: ',x.shape)
    x = Activation("relu")(x)
    # x = Flatten()(x)
    x = Dense(dense_l)(x)
    # print('x shape: ',x.shape)
    x = Activation("sigmoid")(x)
    # print('x shape: ',x.shape)

    return x * x_real

def AUC_draw(true,pred):

    #Truelist = true.detach().numpy()
    #Problist = pred.detach().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    print(roc_auc)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save("./AUC.tiff")
    png1.close()
    #plt.savefig('./AUC.tif')
    #plt.show()

def net(inputs):
    x = Conv2D(32, (2, 2), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = attention(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (2, 2), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = attention(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (2, 2), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = attention(x)
    x = Dropout(0.5)(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)

    x = Dense(8)(x)
    x = Activation("relu")(x)

    return x

# load data
data1 = pd.read_csv('./data/drug1_629.csv')
data2 = pd.read_csv('./data/drug2_629.csv')

X_1_train, X_1_test, X_2_train,X_2_test = train_test_split(data1,data2,test_size=0.2)
print(X_1_train.shape,X_1_test.shape,X_2_train.shape,X_2_test.shape)

y_train = X_1_train['class']
y_test = X_1_test['class']

X_1_train = X_1_train.drop(labels=['class'],axis=1)
X_1_test = X_1_test.drop(labels=['class'],axis=1)

print(X_1_train.shape,X_1_test.shape,X_2_train.shape,X_2_test.shape)
print(y_train.shape,y_test.shape)

X_1_train.reset_index(drop=True,inplace=True)
X_2_train.reset_index(drop=True,inplace=True)
X_1_test.reset_index(drop=True,inplace=True)
X_2_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)

X_1_train = X_1_train.values.reshape([-1, 1, 300,1])
#print(X_1_train[:3])
X_1_test = X_1_test.values.reshape([-1, 1, 300,1])
X_1_train.astype('float64')
X_1_test.astype('float64')
print(X_1_train.shape,X_1_test.shape)
X_2_train = X_2_train.values.reshape([-1, 1, 300,1])
X_2_test = X_2_test.values.reshape([-1, 1, 300,1])
X_2_train.astype('float64')
X_2_test.astype('float64')

y_train.astype('int64')
y_test.astype('int64')

inputA = Input(shape=(1,300,1))
x = net(inputA)
x = Model(inputs=inputA,outputs=x)

inputB = Input(shape=(1,300,1))
y = net(inputB)
y = Model(inputs=inputB,outputs=y)

#combined = concatenate([x.output,y.output])
combined = x.output + y.output

z = Dense(4,activation='relu')(combined)
z = Dense(2,activation='softmax')(z)
#z = keras.utils.to_categorical(z, 2)
print(z.shape)
model = Model(inputs=[x.input,y.input],outputs=z)

#compile the model
opt = Adam(lr=0.0001,decay=1e-4/50)
# opt = Adam(lr=1e-3)
# model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['binary_accuracy'])
y_test_all = y_test
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

# train the model
print("[INFO] training model...")
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),callbacks=[callback]
model.fit([X_1_train, X_2_train], y_train, validation_data=([X_1_test, X_2_test], y_test),
          epochs=500, batch_size=128)
y_pred = model.predict([X_1_test,X_2_test])

y_pred_all = []
for i in range(len(y_pred)):
    y_pred_all.append(y_pred[i][1])

roc_auc = AUC_draw(y_test_all,y_pred_all)

# validation
val_1 = pd.read_csv('data/pred_1.csv')
val_2 = pd.read_csv('data/pred_2.csv')
val1_test = val_1[0:]
val2_test = val_2[0:]

val1_test = val1_test.values.reshape([-1,1,300,1])
val2_test = val2_test.values.reshape([-1,1,300,1])
print(val1_test.shape)


val_pred = model.predict([val1_test,val2_test])
print(val_pred)

# predict
pred1 = pd.read_csv('data/pred_A20.csv')
pred2 = pd.read_csv('data/pred_Flu.csv')


x1_test = pred1[0:]
x2_test = pred2[0:]

x1_test = x1_test.values.reshape([-1,1,300,1])
x2_test = x2_test.values.reshape([-1,1,300,1])
print(x1_test.shape)

preds = model.predict([x1_test,x2_test])
print(preds)




# serialize model to JSON
model_json = model.to_json()
with open("model_629.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_629.h5")
print("Saved model to disk")


