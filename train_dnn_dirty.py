import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from sklearn import metrics

# DATASET = "traces_out/malware_web_mixed_cleaned.csv"
TRAIN = "traces_out/malware_web_dirty_train.csv"
TEST = "traces_out/malware_web_dirty_test.csv"
VAL = "traces_out/malware_web_dirty_val.csv"
TEST_CLEAN = "traces_out/malware_web_test.csv"
MODEL_CHECKPOINT_PATH = "models/malware_web_dirty_mixed_cleaned_dnn.keras"
CSV_LOGGER_PATH = "logs/malware_web_dirty_mixed_cleaned_dnn.csv"
MIN_LR = 1e-6
START_LR = 1e-3
BATCH_SIZE=32
EPOCHS=200

x_test = pd.read_csv(TEST, header=None)
y_test = x_test.iloc[:, -2:]
x_test = x_test.iloc[:, :-2]
print(f"shape of x_test: {x_test.shape}")
print(f"shape of y_test: {y_test.shape}")

x_test_clean = pd.read_csv(TEST_CLEAN, header=None)
y_test_clean = x_test_clean.iloc[:, -2:]
x_test_clean = x_test_clean.iloc[:, :-2]
print(f"shape of x_test: {x_test_clean.shape}")
print(f"shape of y_test: {y_test_clean.shape}")

x_train = pd.read_csv(TRAIN, header=None)
y_train = x_train.iloc[:, -2:]
x_train = x_train.iloc[:, :-2]
print(f"shape of x_train: {x_train.shape}")
print(f"shape of y_train: {y_train.shape}")

x_val = pd.read_csv(VAL, header=None)
y_val = x_val.iloc[:, -2:]
x_val = x_val.iloc[:, :-2]
print(f"shape of x_val: {x_val.shape}")
print(f"shape of y_val: {y_val.shape}")

#
# Build Model (Simple LSTM)
#

model = tf.keras.Sequential([
    layers.Input(shape=(43,), batch_size=1),
    layers.Dense(20, activation="relu"),
    layers.Dense(15, activation="relu"),
    layers.Dense(15, activation="relu"),
    layers.Dense(13, activation="relu"),
    layers.Dense(10, activation="relu"),
    layers.Dense(5, activation="relu"),
    layers.Dense(2, activation='softmax')])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=START_LR),
    metrics=['accuracy'])

print(model.summary())

# save model to file at each epoch callback
checkpoint = ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH, 
                            monitor='val_loss',
                            verbose=1, 
                            save_best_only=True,
                            mode='min')
# learning rate adjustment callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=MIN_LR)
# early stopping callback
early_stop = EarlyStopping(patience=20, monitor='val_loss')
# csv callback
csv_logger = CSVLogger(CSV_LOGGER_PATH)

model.fit(x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop, csv_logger],
    verbose=1,
    validation_data=(x_val, y_val))

# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.metrics_names)
# f1 score
y_pred = model.predict(x_test, batch_size=2048, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

y_test_bool = np.argmax(y_test.values, axis=1)
print(f"f1: {metrics.f1_score(y_pred_bool, y_test_bool, average=None)}")
print(f"precision: {metrics.precision_score(y_test_bool, y_pred_bool, average=None)}")
print(f"recall: {metrics.recall_score(y_test_bool, y_pred_bool, average=None)}")

# reevaluate on clean test set
print("######\n######\nReevaluating on clean test set\n######\n######")
score = model.evaluate(x_test_clean, y_test_clean, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.metrics_names)
# f1 score
y_pred = model.predict(x_test_clean, batch_size=2048, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

y_test_bool = np.argmax(y_test_clean.values, axis=1)
print(f"f1: {metrics.f1_score(y_pred_bool, y_test_bool, average=None)}")
print(f"precision: {metrics.precision_score(y_test_bool, y_pred_bool, average=None)}")
print(f"recall: {metrics.recall_score(y_test_bool, y_pred_bool, average=None)}")