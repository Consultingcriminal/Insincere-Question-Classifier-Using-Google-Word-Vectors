from model import get_embedding_matrix,get_model
from gensim.models import KeyedVectors
import pandas as pd
import os

if __name__ == '__main__':

    embed_size = 300 
    max_features = 30000 
    max_len = 64
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    news_path = 'GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
    
    df = pd.read_csv('Data/clean_ex_folds.csv')

    for fold in range(5):
        embedding_matrix,X_train,X_valid,y_train,y_valid = get_embedding_matrix(df,fold,embeddings_index,max_len,max_features,embed_size)
        model = get_model(embedding_matrix,max_len,max_features,embed_size)
        model.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_valid, y_valid))
        model.save(f'model_{fold}.h5')