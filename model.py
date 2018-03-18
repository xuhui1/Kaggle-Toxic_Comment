
# coding = utf-8
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, Conv2D, GRU
from keras.layers import Dropout, Embedding,Reshape

from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model

def model_bilstm_cnn(sentence_len,embedding_matrix):
    inp = Input(shape=(sentence_len,),dtype=int32)
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.35)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    out = Dense(6, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# the difference of conv1D and conv2D：http://blog.csdn.net/hahajinbu/article/details/79535172
# conv1D: kernel_size=3 -> (3,embed_size)
# conv2D: kernel_size=3 -> (3,3,1)  so define the kernel size=(3,embed_size) -> (3,embed_size,1)
def model_mulitkernel_cnn(sentence_len,embedding_matrix,num_filters=64):
    inp = Input(shape(sentence_len,),dtype=int32)
    x = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((sentence_len,embedding_matrix.shape[1],1))(x)
    
    filter_sizes = [1,2,3,5]
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(sentence_len - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sentence_len - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sentence_len - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(sentence_len - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(6, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def model_lstm_concat_cnn(sentence_len,embedding_matrix,num_filters):
    inp = Input(shape(sentence_len,),dtype=int32)
    x = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)
    
    cnn = Conv1D(num_filters, kernel_size=3, kernel_initializer='normal',activation='elu')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    cnn = concatenate([avg_pool, max_pool])
    
    lstm =  Bidirectional(GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    
    x =  Concatenate(axis=1)([cnn,lstm])
    x = Dropout(0.4)(x)
    outp = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    
def model_lstm_cnn(sentence_len,embedding_matrix):
    inp = Input(shape(sentence_len,),dtype=int32)
    x = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)
    
    x = Reshape((sentence_len,embedding_matrix.shape[1],1))(x)
    x = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    x = Dropout(0.2)(x)
    x_w, x_h, x_d = [int(value) for value in x.shape[1:]]
    x = Permute((1,3,2))(x)
    x = Reshape((x_w, x_h * x_d))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = SpatialDropout1D(0.2)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outp = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    
