from pickle import load
import pandas as pd

def load_and_predict(file,data):
    
    model_cl = load(open(f'models/{file}','rb'))
    dataframe = pd.read_csv(f'datasets/{data}',delimiter=',')
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    
    try:
        
        return dataframe,model_cl.predict(dataframe),model_cl.best_params_,file
    
    except:
        return -1
    
