import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow import keras

#config
img_dimension = 299
total_words = 10000 + 1
encoding_size = 512
LSTM_size = 512
max_cap_len = 15

incep = keras.applications.inception_v3.InceptionV3(input_shape=(img_dimension,img_dimension,3), include_top=False)
incep.trainable=False
incep.summary()


encoder = keras.models.Sequential([
                                   keras.layers.Lambda(preprocess_input,input_shape=(img_dimension,img_dimension,3),name="preprocessing_layer"),
                                   incep,
                                   keras.layers.Dense(encoding_size,activation='relu',name="encoding_layer"),
                                   keras.layers.Reshape((8*8,encoding_size),name="reshape_layer")
],name="Encoder")


encoder.load_weights("encoder.hdf5")
#encoder.summary()


decoder = keras.models.load_model("decoder.hdf5")
#decoder.summary()


import pickle
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (img_dimension, img_dimension))
  return img


def caption_image(path):
  image = load_img(path)#/255.0

  encodings = encoder.predict(tf.reshape(image,(1,img_dimension,img_dimension,3)))

  texts = ["<sos>"]
  h = np.zeros((1,LSTM_size))
  c = np.zeros((1,LSTM_size))
  for _ in range(max_cap_len + 1):
    dec_inp = np.array(tok.word_index.get(texts[-1])).reshape(1,-1)
    #print(dec_inp)
    props,h,c = decoder.predict([encodings,h,c ,dec_inp])
    props= props[0]
    idx = np.argmax(props)
    
    texts.append(tok.index_word.get(idx))
    
    if idx == tok.word_index['<eos>']:
      break
  if tok.word_index.get(texts[-1]) != tok.word_index['<eos>']:
    texts.append('<eos>')
  print(' '.join(texts))
  #plt.imshow(image/255.0)
  #plt.axis("off")
  return " ".join(texts)


img_name = "IMG11.jpg"
caption = caption_image(img_name)

import cv2

def plot_img(img, size=(7, 7), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.axis("off")
    plt.show()


image = cv2.imread(img_name, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plot_img(image, title=caption)