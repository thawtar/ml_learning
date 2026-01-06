import yaml
# data sources might be google drive or local, default is local
config = {
    'data':{
        'source':'local', # local or gdrive
        'preprocess':True, # always preprocess data as long as instructed otherwise
        'local':{
            'data_path':'../data/customer_churn_data.csv'
        },
        'gdrive':{
            'file_id':'1A2B3C4D5E6F7G8H9I0J', # example file id
            'credentials_path':'./config/credentials.json',
            'token_path':'./config/token.json'
        }
    },
    'model':{
        'type':'svm', # svm, random_forest, logistic_regression, xgboost
        'save_path':'./models/churn_model.pkl'
    },

}

def load_yaml(path='config.yaml'):
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_yaml(path='config.yaml'):
    with open(path,'w') as f:
        yaml.dump(config,f)

if __name__ == "__main__":
    print("Data Configuration:")
    yaml_dump = yaml.dump(config, default_flow_style=False)
    print(yaml_dump)