from parameters import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv1D, Conv2D, Flatten, AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import GRU, TimeDistributed, SimpleRNN, Dropout, Average
from tensorflow.keras.layers import MaxPooling1D
from keras.utils.vis_utils import plot_model


def generate_model(type):

    if type == 'CNN':
        model = Sequential([
                Reshape((cx,1), input_shape=(cx,)),
                Conv1D(filters=1, activation=act_function, kernel_size=3, data_format="channels_last",
                strides=1, padding="same", input_shape=(cx,1)),
                #Conv1D(filters=1, activation=act_function, kernel_size=30, data_format="channels_last",
                #strides=1, padding="same"),
                Reshape((cx,))
                ])
    
    if type == 'CNN_2D':
        model = Sequential([
                Reshape((cx,cy,1), input_shape=(cx,cy,)),
                Conv2D(filters=1, activation=act_function, kernel_size=(3,3), data_format="channels_last",
                strides=1, padding="same", input_shape=(cx, cy)),
                Reshape((cx,cy,))
                ])
    
    if type == 'Dense':
        model = Sequential([
                Dense(cx, input_dim = cx, activation=act_function),
                #Dense(1000, activation=act_function),
                #Dense(500, activation=act_function),
                Dense(cx, activation=act_function)
                ])

    if type == 'Dense_2D':
        model = Sequential([
                Reshape((cx*cy,1), input_shape=(cx,cy,)),
                Dense(cx*cy, input_dim = cx*cy, activation=act_function),
                #Dense(1000, activation=act_function),
                #Dense(500, activation=act_function),
                Dense(cx*cy, activation=act_function),
                Reshape((cx,cy,))
                ])

    #if type == 'RNN':
    #    model = 
    
    model.compile(optimizer=opt_method, loss=loss_function, metrics=['accuracy'])
    model.save('./model/model.h5')
    return model 