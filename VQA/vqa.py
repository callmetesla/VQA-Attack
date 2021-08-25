import os
import time
import warnings
import h5py
warnings.filterwarnings("ignore")
import cv2
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import spacy
import keras.backend as backend
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense



def remove_layer(model):
    model.layers.pop()
    if not model.layers:
        model.inbound_nodes = []
        model.outputs = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

    return model


def VGG_16(weights):
    weight_dict = h5py.File(weights, 'r')
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    flattened = model.layers

    naive_bayes_layer = weight_dict.attrs['nb_layers']

    for layer in range(naive_bayes_layer):
        val = weight_dict['layer_{}'.format(layer)]
        w = [val['param_{}'.format(param)] for param in range(val.attrs['nb_params'])]
        if not w : continue
        if len(w[0].shape) >2: 
            w[0] = np.swapaxes(w[0],0,3)
            w[0] = np.swapaxes(w[0],0,2)
            w[0] = np.swapaxes(w[0],1,2)
        flattened[layer].set_weights(w)
    # remove last tow layers to match dimensions
    model = remove_layer(model)
    model = remove_layer(model)
    weight_dict.close()    
    print("VGG MODEL")
    print (model.summary())
    return model


def set_params():
    backend.set_image_data_format('channels_first')
# sets image to be read as (depth, input_depth, rows, cols)
    backend.set_image_dim_ordering('th')


def VQA():
    image_size = 4096
    word_size = 300
    n_lstm= 3
    n_hidden_lstm= 512
    max_word = 30 # max words in question
    n_dense = 3
    n_hidden = 1024
    activation = 'tanh'
    dropout = 0.5
    #Image layer

    image_model= Sequential()
    image_model.add(Reshape((image_size,), input_shape=(image_size,)))
    #LSTM layer
    language_model = Sequential()
    language_model.add(LSTM(n_hidden_lstm, return_sequences=True, input_shape=(max_word, word_size)))
    language_model.add(LSTM(n_hidden_lstm, return_sequences=True))
    language_model.add(LSTM(n_hidden_lstm, return_sequences=False))

    #combine model
    model = Sequential()
    model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))

    for _ in range(n_dense):
        model.add(Dense(n_hidden, kernel_initializer='uniform'))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    model.add(Dense(1000))
    #Final layer with Top 1000 answers
    model.add(Activation('softmax'))
    print (model.summary())
    return model


def main(image_path=None, ques=None):
    start_time = time.time()
    set_params()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("Obtaining image in form of features")
    image_resized = cv2.resize(cv2.imread(image_path), (224, 224))
    image_float  = image_resized.astype(np.float32)
    normalized_vector = [103.939, 116.779, 123.68]
    for third_dimension in range(3):
        image_float[:, :, third_dimension] = image_float[:, :, third_dimension] - normalized_vector[third_dimension]
    # convert from width,height,channel to channel,width,height    
    image_float = image_float.transpose((2,0,1)) 
    print("Image has dimensions"+ str(image_float.shape))
    image = np.expand_dims(image_float, axis=0) 
    # Adding extra dimension for maintaining model size
    print("Image now has dimensions"+ str(image_float.shape))
    im_features = np.zeros((1, 4096))
    model_image = VGG_16('VQA_Attack/Weights/vgg16_dict.h5')
    stochastic_gd= SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_image.compile(optimizer=stochastic_gd, loss='categorical_crossentropy')
    im_features[0,:] = model_image.predict(image)[0]
    print("Converting question into embeddings")
    embedding = spacy.load('en_vectors_web_lg')
    word_as_tokens = embedding(str(ques))
    tensor_q = np.zeros((1, 30, 300))
    # 30 because max is 30 words in a question
    for x in range(len(word_as_tokens)):
        tensor_q[0,x,:] = word_as_tokens[x].vector
    vqa = VQA()
    vqa.load_weights('VQA_Attack/Weights/VQA_weights.hdf5')
    vqa.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("VQA MODEL")
    print (vqa.summary())
    print("Inferencing....................")
    output = vqa.predict([tensor_q, im_features])
    # Answer will be a top 1000 answer vector
    # Answers are encoded as labels to minimize compute as words increase compute
    encoder = joblib.load('VQA_Attack/Weights/labelencoder.pkl')
    answer_vector = {}
    answer_vector['answer']= []
    answer_vector['answer_prob']= []
    temp_list = []
    for label in reversed(np.argsort(output)[0,-5:]):
        temp_list.append(label)
        answer_vector['answer'].append(str(encoder.inverse_transform(temp_list)[0]))
        answer_vector['answer_prob'].append(str(round(output[0,label]*100, 2)))
        temp_list.pop()
    print (answer_vector)
    print("Time taken for prediction: " +str(time.time()-start_time))
    return answer_vector


if __name__ == "__main__":
    main('test.jpg', "What vechile is in the picture?")
