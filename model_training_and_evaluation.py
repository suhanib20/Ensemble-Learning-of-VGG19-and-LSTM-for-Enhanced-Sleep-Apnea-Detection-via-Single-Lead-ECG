# -*- coding: utf-8 -*-
"""VGG19-LSTM.ipynb"""

import pickle
import numpy as np
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Permute, Reshape, LSTM, Dropout, BatchNormalization
from keras.regularizers import l2
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

base_dir = "/content/drive/MyDrive/dataset/osa_data"

ir = 3
time_range = 60
weight = 1e-3

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr)) / (np.max(arr) - np.min(arr))

def load_data():
    tm = np.arange(0, time_range, step=1 / ir)
    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    x = []
    X, Y = apnea_ecg["o_train"], apnea_ecg["y_train"]
    for i in range(len(X)):
        (rri_tm, rri_signal), (amp_tm, amp_signal) = X[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(amp_tm, scaler(amp_signal), k=3), ext=1)
        x.append([rri_interp_signal, amp_interp_signal])
    x = np.array(x, dtype="float32")

    x = np.expand_dims(x, 1)
    x_final = np.array(x, dtype="float32").transpose((0, 3, 1, 2))

    return x_final, Y

def create_model(weight=1e-3):
    model = Sequential()
    model.add(Reshape((90, 2, 2), input_shape=(180, 1, 2)))

    model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((2, 5 * 512)))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(21, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    return model

if __name__ == "__main__":
    X, Y = load_data()
    Y = tf.keras.utils.to_categorical(Y, num_classes=2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    ensemble_size = 5
    models = []
    histories = []

    for _ in range(ensemble_size):
        for train, test in kfold.split(X, Y.argmax(1)):
            model = create_model()
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            X1, x_val, Y1, y_val = train_test_split(X[train], Y[train], test_size=0.10)

            history = model.fit(X1, Y1, batch_size=130, epochs=100, validation_data=(x_val, y_val),
                                callbacks=[early_stopping, lr_scheduler])

            histories.append(history)
            models.append(model)

    def ensemble_predict(models, X):
        predictions = [model.predict(X) for model in models]
        return np.mean(predictions, axis=0)

    train_accuracies = []
    test_accuracies = []

    for train, test in kfold.split(X, Y.argmax(1)):
        y_train_pred = ensemble_predict(models, X[train])
        y_test_pred = ensemble_predict(models, X[test])

        train_accuracy = np.mean(np.argmax(y_train_pred, axis=-1) == np.argmax(Y[train], axis=-1))
        test_accuracy = np.mean(np.argmax(y_test_pred, axis=-1) == np.argmax(Y[test], axis=-1))

        train_accuracies.append(train_accuracy * 100)
        test_accuracies.append(test_accuracy * 100)

    print("Ensemble model train accuracies: ", train_accuracies)
    print("Mean train accuracy: ", np.mean(train_accuracies))
    print("Ensemble model test accuracies: ", test_accuracies)
    print("Mean test accuracy: ", np.mean(test_accuracies))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for history in histories:
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    for history in histories:
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show() #first file sending the next one 

