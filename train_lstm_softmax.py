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

DATASET = "traces_out/malware_web_mixed_cleaned.csv"
MODEL_CHECKPOINT_PATH = "models/malware_web_mixed_cleaned_softmax.keras"
CSV_LOGGER_PATH = "logs/malware_web_mixed_cleaned_softmax.csv"
MIN_LR = 1e-6
START_LR = 1e-3
BATCH_SIZE=32
EPOCHS=200

X = pd.read_csv(DATASET, header=None)
Y = X.iloc[:, -2:]
X = X.iloc[:, :-2]
print(f"shape of X: {X.shape}")
print(f"shape of Y: {Y.shape}")

#
# Build Model (Simple LSTM)
#

model = tf.keras.Sequential([
    layers.Input(shape=(43,), batch_size=1),
    layers.Embedding(256, 32),
    layers.LSTM(16),
    layers.Dense(2, activation='softmax')])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=START_LR),
    metrics=['accuracy'])

print(model.summary())

# save model to file at each epoch callback
checkpoint = ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH, 
                            monitor='loss',
                            verbose=1, 
                            save_best_only=True,
                            mode='min')
# learning rate adjustment callback
reduce_lr = ReduceLROnPlateau(monitor='loss', min_lr=MIN_LR)
# early stopping callback
early_stop = EarlyStopping(patience=20, monitor='loss')
# csv callback
csv_logger = CSVLogger(CSV_LOGGER_PATH)

model.fit(X, Y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop, csv_logger],
    verbose=1)