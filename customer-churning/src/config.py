# data sources might be google drive or local, default is local
data_conf = {
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
}