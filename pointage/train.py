import mtcnn
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from pointage.constant import PATH_FACENET_MODEL, PATH_DATASET
from pointage.utils import data_processing
import numpy as np
from sklearn.metrics import accuracy_score

detectors = mtcnn.MTCNN()
MODEL_FACENET = load_model(PATH_FACENET_MODEL)

def train():

    y_train, x_train = data_processing(PATH_DATASET)

    # Feature extraction by facenet
    x_train = MODEL_FACENET.predict(np.array(x_train))

    # Encode data
    input_encoder = Normalizer(norm='l2')
    out_encoder = LabelEncoder()
    x_train = input_encoder.fit_transform(x_train)
    y_train = out_encoder.fit_transform(y_train)

    # Train data with models
    model_svm = SVC(kernel='linear', probability=True)
    model_svm.fit(x_train, y_train)

    model_svm_rbf = SVC(kernel='rbf', probability=True)
    model_svm_rbf.fit(x_train, y_train)

    model_svm_poly = SVC(kernel='poly', degree=3, probability=True)
    model_svm_poly.fit(x_train, y_train)

    # Predict model
    yhat_train = model_svm.predict(x_train)
    yhat_train_rbf = model_svm_rbf.predict(x_train)
    yhat_train_poly = model_svm_poly.predict(x_train)

    # Evaluate model
    score_train = accuracy_score(yhat_train, y_train)
    score_train_poly = accuracy_score(y_train, yhat_train_poly)
    score_train_rbf = accuracy_score(y_train, yhat_train_rbf)

    return model_svm_poly
