from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_dataset(df,dataset_parameters):

    df = df.drop(['Unnamed: 0', 'Block Name', 'Point ID', 'CoordinateX', 
                'Points_0','Points_1', 'Points_2', 'Points_Magnitude', 'Result_0', 'Result_1',
        'Result_2', 'Result_Magnitude'], axis=1)

    train_df, test_df = train_test_split(df, test_size=dataset_parameters.get('test_size'), random_state=1)

    train_X, train_y = train_df.iloc[:,2:-1].values,train_df.iloc[:,-1].values
    train_y=train_y.reshape(-1,1)
    test_X,test_y = test_df.iloc[:,2:-1].values,test_df.iloc[:,-1].values
    test_y=test_y.reshape(-1,1)

    over = SMOTE(sampling_strategy=dataset_parameters.get('sampling_strategy'))
    train_X, train_y = over.fit_resample(train_X, train_y)

    return train_X, train_y, test_X, test_y
