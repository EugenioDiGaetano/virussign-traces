import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn import metrics

TEST = "traces_out/malware_web_dirty_test.csv"
TEST_CLEAN = "traces_out/malware_web_test.csv"
MODEL_CHECKPOINT_PATH_DIRTY = "models/malware_web_dirty_mixed_cleaned_dnn.keras"
MODEL_CHECKPOINT_PATH = "models/malware_web_mixed_cleaned_dnn.keras"


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


# load best model from file
model = keras.models.load_model(MODEL_CHECKPOINT_PATH)

# evaluate model
score = model.evaluate(x_test_clean, y_test_clean, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.metrics_names)
# f1 score
y_pred = model.predict(x_test_clean, batch_size=2048, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

y_test_bool = np.argmax(y_test_clean.values, axis=1)
print(f"f1: {metrics.f1_score(y_pred_bool, y_test_bool, average=None)}")
print(f"avg: {metrics.f1_score(y_test_bool, y_pred_bool)}")
print(f"precision: {metrics.precision_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.precision_score(y_test_bool, y_pred_bool)}")
print(f"recall: {metrics.recall_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.recall_score(y_test_bool, y_pred_bool)}")

print("######\n######\n######\nReevaluating on dirty DNN\n######\n######\n######")
# load best model from file
model = keras.models.load_model(MODEL_CHECKPOINT_PATH_DIRTY)

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
print(f"avg: {metrics.f1_score(y_test_bool, y_pred_bool)}")
print(f"precision: {metrics.precision_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.precision_score(y_test_bool, y_pred_bool)}")
print(f"recall: {metrics.recall_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.recall_score(y_test_bool, y_pred_bool)}")

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
print(f"avg: {metrics.f1_score(y_test_bool, y_pred_bool)}")
print(f"precision: {metrics.precision_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.precision_score(y_test_bool, y_pred_bool)}")
print(f"recall: {metrics.recall_score(y_test_bool, y_pred_bool, average=None)}")
print(f"avg: {metrics.recall_score(y_test_bool, y_pred_bool)}")