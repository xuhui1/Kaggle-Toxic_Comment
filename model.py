
# coding = utf-8
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding

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
