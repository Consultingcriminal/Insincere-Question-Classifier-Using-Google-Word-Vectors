import pandas as pd
import numpy as np
import re

def clean_text(x):

    x = str(x)
    x=re.sub(r'\<br /><br />\b\s+',"",x)
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!"#$%\'()*+-/:;<=>@[\\]^_`{|}~.,' + '“”’':
        x = x.replace(punct, '') 
   
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def cont_to_exp(x):
    contractions = { 
                        "ain't": "am not",
                        "aren't": "are not",
                        "can't": "cannot",
                        "can't've": "cannot have",
                        "'cause": "because",
                        "could've": "could have",
                        "couldn't": "could not",
                        "couldn't've": "could not have",
                        "didn't": "did not",
                        "doesn't": "does not",
                        "don't": "do not",
                        "hadn't": "had not",
                        "hadn't've": "had not have",
                        "hasn't": "has not",
                        "haven't": "have not",
                        "he'd": "he had",
                        "he'd've": "he would have",
                        "he'll": "he will",
                        "he'll've": "he will have",
                        "he's": " he is",
                        "how'd": "how did",
                        "how'd'y": "how do you",
                        "how'll": "how will",
                        "how's": "how has",
                        "I'd": "I would",
                        "I'd've": "I would have",
                        "I'll": "I will",
                        "I'll've": "I will have",
                        "I'm": "I am",
                        "I've": "I have",
                        "isn't": "is not",
                        "it'd": "it had",
                        "it'd've": "it would have",
                        "it'll": "it will",
                        "it'll've": "it will have",
                        "it's": "it is",
                        "let's": "let us",
                        "ma'am": "madam",
                        "mayn't": "may not",
                        "might've": "might have",
                        "mightn't": "might not",
                        "mightn't've": "might not have",
                        "must've": "must have",
                        "mustn't": "must not",
                        "mustn't've": "must not have",
                        "needn't": "need not",
                        "needn't've": "need not have",
                        "o'clock": "of the clock",
                        "oughtn't": "ought not",
                        "oughtn't've": "ought not have",
                        "shan't": "shall not",
                        "sha'n't": "shall not",
                        "shan't've": "shall not have",
                        "she'd": "she had",
                        "she'd've": "she would have",
                        "she'll": "she will",
                        "she'll've": "she will have",
                        "she's": "she is",
                        "should've": "should have",
                        "shouldn't": "should not",
                        "shouldn't've": "should not have",
                        "so've": "so have",
                        "so's": "so as",
                        "that'd": "that would",
                        "that'd've": "that would have",
                        "that's": "that has",
                        "there'd": "there had",
                        "there'd've": "there would have",
                        "there's": "there is",
                        "they'd": "they had",
                        "they'd've": "they would have",
                        "they'll": "they will",
                        "they'll've": "they will have",
                        "they're": "they are",
                        "they've": "they have",
                        "to've": "to have",
                        "wasn't": "was not",
                        "we'd": "we had",
                        "we'd've": "we would have",
                        "we'll": "we will",
                        "we'll've": "we will have",
                        "we're": "we are",
                        "we've": "we have",
                        "weren't": "were not",
                        "what'll": "what will",
                        "what'll've": "what will have",
                        "what're": "what are",
                        "what's": "what is",
                        "what've": "what have",
                        "when's": "when is",
                        "when've": "when have",
                        "where'd": "where did",
                        "where's": "where is",
                        "where've": "where have",
                        "who'll": "who will",
                        "who'll've": "who will have",
                        "who's": "who is",
                        "who've": "who have",
                        "why's": "why is",
                        "why've": "why have",
                        "will've": "will have",
                        "won't": "will not",
                        "won't've": "will not have",
                        "would've": "would have",
                        "wouldn't": "would not",
                        "wouldn't've": "would not have",
                        "y'all": "you all",
                        "y'all'd": "you all would",
                        "y'all'd've": "you all would have",
                        "y'all're": "you all are",
                        "y'all've": "you all have",
                        "you'd": "you would",
                        "you'd've": "you would have",
                        "you'll": " you will",
                        "you'll've": "you will have",
                        "you're": "you are",
                        "you've": "you have"
                        } 
    if type(x) is str:
        for key in contractions:
            value=contractions[key]
            x=x.replace(key,value)
        return x
    else:
        return x    

def text_preprocess(df,target):

    df[target] = df[target].apply(lambda x:x.lower())
    df[target] = df[target].apply(lambda x:cont_to_exp(x))
    df[target] = df[target].apply(lambda x:clean_text(x))
    df[target] = df[target].apply(lambda x:clean_numbers(x))
    return df    

if __name__ == '__main__':
    df=pd.read_csv('Data/train.csv',low_memory=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df_1=df[df['target']==1].head(10000)
    df_0=df[df['target']==0].head(30000)
    df=df_1.append(df_0,ignore_index=True)
    cleaned_df=text_preprocess(df,'question_text')
    print(cleaned_df.head())
    cleaned_df.to_csv('Data/cleaned_text.csv',index=False)
