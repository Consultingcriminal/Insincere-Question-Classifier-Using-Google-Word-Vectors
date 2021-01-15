import pandas as pd 
from sklearn import model_selection

def create_folds(df):
    df['question_text']=df['question_text']
    df.loc[:,"kfold"]=-1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target.values
    skf = model_selection.StratifiedKFold(n_splits=5)


    for f, (t_, v_) in enumerate(skf.split(X=df,y=y)):
        df.loc[v_,"kfold"] = f

    return df

if __name__ =="__main__":
    df = pd.read_csv('Data/extended_text.csv')
    df2 = create_folds(df)
    print(df2.head())
    df2.to_csv('Data/clean_ex_folds.csv',index=False)