from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Conv1D
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics

def get_embedding_matrix(df,fold,embeddings_index,max_len=128,max_features=30000,embed_size=300):
    df_train=df[df.kfold != fold].reset_index(drop=True)
    df_valid=df[df.kfold == fold].reset_index(drop=True)
    
    #Tokenizing
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(df_train['question_text']))
    X_train = tokenizer.texts_to_sequences(df_train['question_text'])
    X_valid = tokenizer.texts_to_sequences(df_valid['question_text'])
    
    #Padding
    X_train = pad_sequences(X_train, maxlen=max_len,truncating='post')
    X_valid = pad_sequences(X_valid, maxlen=max_len,truncating='post')
    
    #Target Values
    y_train = df_train.target.values
    y_valid = df_valid.target.values
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
    for word, i in word_index.items():
        if i >= max_features: continue
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix[i] = embedding_vector
    return embedding_matrix,X_train,X_valid,y_train,y_valid

def get_model(embedding_matrix,max_len,max_features,embed_size):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = CuDNNGRU(64, return_sequences=True)(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.Adam(learning_rate=0.005),
                  metrics=['AUC'])

    return model 