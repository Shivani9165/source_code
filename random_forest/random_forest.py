import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from mycolorpy import colorlist as mcp
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from data_preprocessing.data_preparation import *

def rf_train_model(train_dataset_path,train_hyperparameters):

    dataset_parameters = {}
    dataset_parameters['test_size'] = train_hyperparameters.get('test_size')
    dataset_parameters['sampling_strategy'] = train_hyperparameters.get('sampling_strategy')

    df = pd.read_csv(train_dataset_path)
    train_X, train_y, test_X, test_y = train_dataset(df,dataset_parameters)

    model = RandomForestClassifier(n_estimators=train_hyperparameters.get('n_estimators'))
    model.fit(train_X,train_y)
    pickle.dump(model, open('C:\\vortex_framework-main\\data\\trained_models\\ensembled\\time_step_57.pkl', 'wb'))

    out= model.predict(test_X)
    out=out.reshape(-1,1)

    model_summary ={}
    model_summary['accuracy'] = accuracy_score(test_y,out)*100
    model_summary['precision'] = precision_score(test_y, out,  average="macro")
    model_summary['recall'] = recall_score(test_y, out,  average="macro")
    model_summary['f1_score'] = f1_score(test_y, out,  average="macro")
    model_summary['tn'], model_summary['fp'], model_summary['fn'], model_summary['tp'] = confusion_matrix(test_y, out).ravel()
    
    return model_summary
    
    
def rf_predict_model(dataset_path, model_path,k_value):

    time_step=os.path.basename(dataset_path).split('.')[0].split('_')[2]
    # print(time_step)
    df = pd.read_csv(dataset_path)
    df = df.drop(['Block Name', 'Point ID', 'CoordinateX', 
                'Points_0','Points_1', 'Points_2', 'Points_Magnitude', 'Result_0', 'Result_1',
        'Result_2', 'Result_Magnitude'], axis=1)

    test_X= df.iloc[:,2:].values

    model = pickle.load(open(model_path, 'rb'))
    
    out= model.predict(test_X)
    out=out.reshape(-1,1)

    new_df = np.concatenate((df,out), axis=1)
    # new_test_df = scaler.inverse_transform(new_test_df)
    new_df=pd.DataFrame(new_df,columns=['CoordinateY','CoordinateZ','Magnitude_Velocity_Dataset_V_MAG_N_1_1_0',
                                                'Magnitude_Vorticity_Dataset_VORTICITY_MAG_N_1_1_0',
                                                    'X_Component_Velocity_Dataset_V_X_N_1_1_0',
                                                    'Y_Component_Velocity_Dataset_V_Y_N_1_1_0',
                                                    'Z_Component_Velocity_Dataset_V_Z_N_1_1_0', 'Vortex'])
    new_df = new_df[new_df['Vortex']==1]
    x = new_df[['CoordinateY','CoordinateZ']]

    kmeans = KMeans(k_value)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)

    x['Clusters'] = identified_clusters 
    centroids  = kmeans.cluster_centers_
    centroid_labels = [centroids[i] for i in identified_clusters]

    array_x = np.asarray(x.iloc[:,:2])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, array_x)

    array_x_list = [array_x[x] for x in closest]
    cluster_label = ['Vortex: '+str(identified_clusters[x]) for x in closest]
    cluster_coordinates=[list(array_x[x]) for x in closest]
    array_x_list_x = [x[0] for x in array_x_list]
    array_x_list_y = [x[1] for x in array_x_list]
    vortex_core_dict=dict(zip(cluster_label,cluster_coordinates))

    color_list=mcp.gen_color(cmap="rainbow",n=k_value)
    for i in range(k_value):
        subset_x = x[x['Clusters']==i]
        plt.scatter(subset_x['CoordinateY'],subset_x['CoordinateZ'],c=color_list[i],label='Vortex: '+str(i))
    plt.scatter(array_x_list_x,array_x_list_y,c='black')
    plt.xlabel('Y-Coordinate')
    plt.ylabel('Z-Coordinate')
    plt.legend(bbox_to_anchor=(1.3,0), loc="lower right")
    plt.savefig('C:\\vortex_framework-main\\data\\model_prediction\\random_forest_model\\time_step_'+str(time_step)+'_vortex_plot.png',bbox_inches='tight')

    return vortex_core_dict