import numpy as np
import os
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from module.layers.Dense import Dense
from module.Sequential import Sequential
from module.optimizer.Adam import Adam
from module.layers.RNN import RNN
from tensorflow import keras
from module.layers.Flatten import Flatten


os.system("cls")


def get_wv(w):
    try:
        return w2v.wv[w]
    except KeyError:
        return w2v.wv["UNK"]

df = pd.read_csv("./dataset/DanhgiaSmartphone.csv")
sentences = df["comment"].values
words = [[word for word in sen.lower().split()] for sen in sentences]
n = len(sentences)
y = OneHotEncoder(sparse_output=False).fit_transform(df["label"].values.reshape(-1, 1))
# [001] pos
# [100] neg
# [010] neu

# Chuyen word sang vector
#w2v = Word2Vec(words, vector_size=5)
#unk_vector = np.random.randn(w2v.vector_size)
#w2v.wv.add_vector("UNK", unk_vector)
#w2v.save("test.model") 
w2v = Word2Vec.load("test.model")

X = []
for word in words:
    data = []
    for w in word:
        data.append(get_wv(w))
    X.append(data)

X = keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype="float32")


""" (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X = x_train[:1000]
y = OneHotEncoder(sparse_output=False).fit_transform(y_train[:1000].reshape(-1, 1)) """

md = Sequential()
md.add(RNN(64, active="relu"))
md.add(Dense(3, active="softmax"))
md.compile(optimizer=Adam(lr=0.01, beta1=0.9, beta2=0.999))
md.fit(X=X, y=y, batch_size=5, epochs=3)
