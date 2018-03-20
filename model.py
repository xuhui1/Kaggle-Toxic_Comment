
# coding = utf-8
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, Conv2D, GRU,Lambda
from keras.layers import Dropout, Embedding,Reshape

from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model

from keras.backend import K

def char_word_bilstm_cnn(sentence_len,embedding_matrix,char_vocab_size,char_embedding_size=25):
    #word embedding
    word_ids = Input(batch_shape=(None, None), dtype='int32')
    word_embeddings = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],trainable=True)(word_ids)
    #char embedding
    char_ids = Input(batch_shape=(None, None, None), dtype='int32')
    char_embeddings = Embedding(input_dim=char_vocab_size,output_dim=char_embedding_size,mask_zero=True )(char_ids)
    
    s = K.shape(char_embeddings)
    char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_embedding_size)))(char_embeddings)
    
    fwd_state = LSTM(25, return_state=True)(char_embeddings)[-2]
    bwd_state = LSTM(25, return_state=True, go_backwards=True)(char_embeddings)[-2]
    char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
    # shape = (batch size, max sentence length, char hidden size)
    char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 25]))(char_embeddings)
    
    x = Concatenate(axis=-1)([word_embeddings, char_embeddings])
    x = Dropout(config.dropout)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    out = Dense(6, activation='sigmoid')(x)

    sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
    
    model = Model(inputs=[word_ids, char_ids, sequence_lengths], out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

def model_lstm_concat_cnn(embedding_matrix,sentence_len, num_filters=64):
    inp = Input(shape=(sentence_len,), dtype='int32')
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)

    cnn = Conv1D(num_filters, kernel_size=3, kernel_initializer='normal', activation='elu')(x)
    # avg_pool = GlobalAveragePooling1D()(cnn)
    cnn = MaxPooling1D(pool_size=6)(cnn)
    # cnn = concatenate([avg_pool, max_pool])
    cnn = Flatten()(cnn)

    lstm = Bidirectional(GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)

    x = Concatenate(axis=1)([cnn, GlobalAveragePooling1D()(lstm),GlobalMaxPooling1D()(lstm)])
    x = Dropout(0.4)(x)
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    
def model_cnn_lstm(sentence_len,embedding_matrix):
    inp = Input(shape=(sentence_len,),dtype='int32')
    x = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)
    
    x = Reshape((sentence_len,embedding_matrix.shape[1],1))(x)
    x = Conv2D(num_filters, kernel_size=(3, embedding_matrix.shape[1]), kernel_initializer='normal',
                                                                                    activation='relu')(x)
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
    
def model_depCNN(sentence_len,embedding_matrix):
    comment_input = Input(shape=(sentence_len,), dtype='int32')
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input) 
    
    kernel_size = 3
    #activation_func = LeakyReLU()
    activation_func = Activation('relu')

    # Convolutional block_1
    conv1 = Conv1D(64, kernel_size)(embedded_sequences) #64
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

    conv4 = Conv1D(256, kernel_size)(pool1)
    act4 = activation_func(conv4)
    bn4 = BatchNormalization()(act4)
    pool4 = MaxPooling1D(pool_size=4, strides=4)(bn4)    
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(pool4)
    gmeanpl = GlobalAveragePooling1D()(pool4)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)
    dense1 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(mergedlayer)
    actmlp = activation_func(dense1) 
    reg = Dropout(0.5)(actmlp)
    
    dense2 = Dense(6, activation='softmax')(reg)
    
    model = Model(inputs=[comment_input], outputs=[dense2])
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])  
    return model
