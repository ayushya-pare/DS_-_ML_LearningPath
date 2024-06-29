import pandas as pd

def download_data(file_path):
    return pd.read_csv(file_path, index_col=0)

def basic_exploration(df):
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.head(5))
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())

def advanced_exploration(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(df[numerical_cols].describe())
    for col in categorical_cols:
        print(df[col].value_counts())
    return numerical_cols, categorical_cols
