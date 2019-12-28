import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


def readData(colorType):
    datapath = 'E:/College Crap/MachineLearning/Project/Sign-Language-Digits-Dataset-master/Dataset'
    categories = 10
    data = []
    x = []
    y = []
    avgs = []
    for i in range(0, categories):
        currentNumPath = os.path.join(datapath, str(i))
        for img in os.listdir(currentNumPath):
            if colorType == 1:
                imgArray = cv2.imread(os.path.join(currentNumPath, img), cv2.IMREAD_GRAYSCALE)
            else:
                imgArray = cv2.imread(os.path.join(currentNumPath, img))
                avgs.append(imgArray.mean())

            imgArray = cv2.resize(imgArray, (100, 100))
            data.append([imgArray, i])

    random.shuffle(data)
    avgs = np.array(avgs).mean()

    for features, label in data:
        x.append(features)
        y.append(label)

    if colorType == 1:
        x = np.array(x).reshape(-1, 100, 100, 1)
    else:
        x = np.array(x).reshape(-1, 100, 100, 3)
        x = np.subtract(x, avgs)

    x = np.divide(x, 255)

    return x, y

def createNNModels(x):

    ###########################
    # First model
    ###########################

    model1 = Sequential()

    model1.add(Flatten(input_shape=x.shape[1:]))
    model1.add(Dense(200))
    model1.add(Activation('relu'))

    model1.add(Dense(128))
    model1.add(Activation('relu'))

    model1.add(Dense(10))
    model1.add(Activation('softmax'))

    model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###########################
    # Second model
    ###########################

    model2 = Sequential()

    model2.add(Flatten(input_shape=x.shape[1:]))
    model2.add(Dense(256))
    model2.add(Activation('relu'))

    model2.add(Dense(128))
    model2.add(Activation('relu'))

    model2.add(Dense(64))
    model2.add(Activation('relu'))

    model2.add(Dense(10))
    model2.add(Activation('softmax'))

    model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###########################
    # Third model
    ###########################


    model3 = Sequential()

    model3.add(Flatten(input_shape=x.shape[1:]))

    model3.add(Dense(128))
    model3.add(Activation('relu'))


    model3.add(Dense(10))
    model3.add(Activation('softmax'))

    model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###########################
    # Fourth model
    ###########################


    model4 = Sequential()

    model4.add(Flatten(input_shape=x.shape[1:]))

    model4.add(Dense(10))
    model4.add(Activation('softmax'))

    model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model1, model2, model3, model4


def createModels(x):
    ###########################
    # First model
    ###########################

    model1 = Sequential()

    model1.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Conv2D(32, (3, 3)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Flatten())
    model1.add(Dense(32))

    model1.add(Dense(10))
    model1.add(Activation('softmax'))

    model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    ###########################
    # Second model
    ###########################

    model2 = Sequential()

    model2.add(Conv2D(64, (4, 4), input_shape=x.shape[1:]))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Activation('elu'))

    model2.add(Conv2D(64, (4, 4)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Activation('elu'))

    model2.add(Conv2D(64, (4, 4)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Activation('elu'))

    model2.add(Flatten())
    model2.add(Dense(64))

    model2.add(Dense(10))
    model2.add(Activation('softmax'))

    model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###########################
    # Third model
    ###########################

    model3 = Sequential()

    model3.add(Conv2D(50, (4, 4), input_shape=x.shape[1:]))
    model3.add(MaxPooling2D(pool_size=(3, 3)))
    model3.add(Activation('elu'))

    model3.add(Conv2D(50, (3, 3)))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Activation('relu'))

    model3.add(Flatten())
    model3.add(Dense(64))

    model3.add(Dense(10))
    model3.add(Activation('softmax'))

    model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###########################
    # Fourth model
    ###########################

    model4 = Sequential()

    model4.add(Conv2D(64, (4, 4), input_shape=x.shape[1:]))
    model4.add(MaxPooling2D(pool_size=(3, 3)))
    model4.add(Activation('relu'))

    model4.add(Conv2D(64, (3, 3)))
    model4.add(Activation('relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2)))

    model4.add(Conv2D(64, (2, 2)))
    model4.add(Activation('relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2)))

    model4.add(Flatten())
    model4.add(Dense(32))

    model4.add(Dense(10))
    model4.add(Activation('softmax'))

    model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model1, model2, model3, model4


def main():
    ###############################
    # Grayscale Image Testing
    ###############################

    # Read the data as grayscale
    x, y = readData(1)

    # Read the data as RGB
    x2, y2 = readData(2)



    # Split the data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8)

    X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y2, train_size=0.8)

    # Create Models
    models = createNNModels(x)
    rgbModels = createModels(x2);
    print("Models Fitting")
    print("--------------------------------------")
    for m in range (0, len(models)):
        print("Grayscale Model no " + str(m + 1) + ": ")

        models[m].fit(X_train, Y_train, batch_size=16, validation_split=0.2, epochs=50)
        print("RGB Model no " + str(m + 1) + ": ")
        rgbModels[m].fit(X2_train, Y2_train, batch_size=16, validation_split=0.2, epochs=8)


    print("Models performance on grayscale images")
    print("--------------------------------------")

    for m in range(0, len(models)):
        print("Grayscale Model no " + str(m+1) + ": ")

        predictions = models[m].predict_classes(X_test)
        acc = accuracy_score(y_true=Y_test, y_pred=predictions)
        print("Accuracy: " + str(acc))
        print(classification_report(Y_test, predictions))
        print("========================================")

    print("Models performance on RGB images")
    print("--------------------------------------")

    for m in range(0, len(rgbModels)):
        print("RGB Model no " + str(m + 1) + ": ")
        predictions = rgbModels[m].predict_classes(X2_test)
        acc = accuracy_score(y_true=Y2_test, y_pred=predictions)
        print("Accuracy: " + str(acc))
        print(classification_report(Y2_test, predictions))
        print("========================================")


if __name__ == "__main__":
    main()
