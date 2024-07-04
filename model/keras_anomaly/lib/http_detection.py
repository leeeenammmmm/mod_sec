import sys
import os
import numpy as np
from time import gmtime, strftime
from keras_anomaly_detection.library.recurrent import BidirectionalLstmAutoEncoder
from keras.preprocessing import sequence

os.chdir(sys.path[0])


class HttpPredict():
    def _init_(self):
        print("loaded model")

    # Load Model
    def loadModelBinary(self, model):
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        self.ae = BidirectionalLstmAutoEncoder()
        self.ae.load_model(model)
        print("Loaded model from disk")

    # Load Model
    # Preprocess
    def preprocess(self, data):
        x = [[self.valid_chars[i] for i in data]]
        x = sequence.pad_sequences(x, maxlen=self.maxlen)
        return x

    # Preprocess
    def predict_binary(self, data):
        data = self.preprocess(data)
        print(data)
        print("_____________________________________________")
        print(np.expand_dims(data, axis=2))
        # print(self.ae.threshold)
        anomaly_predict = self.ae.predict(data)[0]
        # print(anomaly_predict)
        # print(self.ae.threshold)
        if anomaly_predict - self.ae.threshold > 100:
            print("case1")
            predict_score = 1
        elif self.ae.threshold - anomaly_predict > 100:
            predict_score = 0
        else:
            predict_score = float(
                abs(anomaly_predict - self.ae.threshold + 100)) / 200
        print(predict_score)
        return predict_score

    def loadModelInit(self):
        self.valid_chars = {' ': 1, '\xa3': 2, '$': 3, '(': 4, '0': 5, '4': 6, '8': 7,
                            '<': 8, '@': 9, '\xc3': 10, 'D': 11, 'H': 12, 'L': 13, 'P': 14, 'T': 15, 'X': 16,
                            'd': 17, 'h': 18, 'l': 19, 'p': 20, 't': 21, 'x': 22, '\xa0': 23, '#': 24, "'": 25,
                            '+': 26, '/': 27, '3': 28, '7': 29, '\xb8': 30, ';': 31, '?': 32, 'C': 33, 'G': 34,
                            'K': 35, 'O': 36, 'S': 37, 'W': 38, '[': 39, '_': 40, '\xe0': 41, 'c': 42, 'g': 43,
                            'k': 44, 'o': 45, 's': 46, 'w': 47, '\x85': 48, '"': 49, '&': 50, '.': 51, '\xb1': 52,
                            '2': 53, '6': 54, '\xb9': 55, ':': 56, '>': 57, 'B': 58, 'F': 59, 'J': 60, 'N': 61,
                            'R': 62, 'V': 63, 'Z': 64, 'b': 65, 'f': 66, 'j': 67, 'n': 68, 'r': 69, 'v': 70,
                            'z': 71, '~': 72, '\x96': 73, '!': 74, '%': 75, ')': 76, '-': 77, '1': 78, '5': 79,
                            '9': 80, '=': 81, 'A': 82, '\xc2': 83, 'E': 84, 'I': 85, 'M': 86, 'Q': 87, 'U': 88,
                            'Y': 89, ']': 90, 'a': 91, 'e': 92, 'i': 93, 'm': 94, 'q': 95, 'u': 96, 'y': 97}

        self.maxlen = 301
        model_dir_path = './models'
        self.loadModelBinary(model_dir_path)
