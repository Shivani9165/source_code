from ensembled_model.ensembled_model import *
from svm.svm import *
from logistic_regression.lr_model import *
from random_forest.random_forest import *
from vision_framework.VortexDetection import _main
import glob
import argparse


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-vortex', type=str, nargs=2, action="store", help='A required string positional argument to predict the model')


args = parser.parse_args()
print(args.vortex)
if args.vortex:
    if args.vortex[0]=='train':
        if args.vortex[1]=='ensemble':
            train_dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\training_dataset_path\\vortex_57_train.csv'
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # ensemble model list
            train_hyperparameters['model_list']=['rfc','svc','xgb']
            # random forest parameters
            train_hyperparameters['n_estimators']=200
            # svm parameters
            train_hyperparameters['kernel']='poly'
            train_hyperparameters['C']=1
            train_hyperparameters['gamma']='auto'
            # xgboost parameters
            train_hyperparameters['learning_rate']=0.6
            train_hyperparameters['max_depth']=2
            train_hyperparameters['n_estimators']=200
            train_hyperparameters['subsample']=0.6
            train_hyperparameters['objective']='binary:logistic'

            model_summary = ensembled_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='svm':
            train_dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\training_dataset_path\\vortex_57_train.csv'
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # svm parameters
            train_hyperparameters['kernel']='poly'
            train_hyperparameters['C']=1
            train_hyperparameters['gamma']='auto'

            model_summary = svm_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='lr':
            train_dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\training_dataset_path\\vortex_57_train.csv'
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # svm parameters
            train_hyperparameters['penalty']='l2'
            train_hyperparameters['C']=40
            train_hyperparameters['solver']='lbfgs'

            model_summary = lr_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1] == 'rf':
            train_dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\training_dataset_path\\vortex_57_train.csv'
            train_hyperparameters= {}

            # random forest parameters
            # dataset preparation parameters
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            train_hyperparameters['n_estimators']=200

            model_summary = rf_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)
            
    elif args.vortex[0]=='predict':

        if args.vortex[1]=='ensemble':
            dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\prediction_dataset_path\\time_step_57.csv'
            model_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\trained_models\\ensembled\\time_step_57.pkl'
            image_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\\Data\\img57.png'
            vision_model_path = ''
            no_of_bounding_boxes=_main.CVMain(image_path)
            vortex_core = ensembled_predict_model(dataset_path,model_path,no_of_bounding_boxes)
            print(vortex_core)

        elif args.vortex[1]=='svm':
            dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\prediction_dataset_path\\time_step_57.csv'
            model_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\trained_models\\svm\\time_step_57.pkl'
            image_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\\Data\\img57.png'
            vision_model_path = ''
            no_of_bounding_boxes=_main.CVMain(image_path)
            vortex_core = svm_predict_model(dataset_path,model_path,no_of_bounding_boxes)
            print(vortex_core)

        elif args.vortex[1]=='lr':
            dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\prediction_dataset_path\\time_step_57.csv'
            model_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\trained_models\\logistic_regression\\time_step_57.pkl'
            image_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\\Data\\img57.png'
            vision_model_path = ''
            no_of_bounding_boxes=_main.CVMain(image_path)
            vortex_core = lr_predict_model(dataset_path,model_path,no_of_bounding_boxes)
            print(vortex_core)

        elif args.vortex[1]=='rf':
            dataset_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\prediction_dataset_path\\time_step_57.csv'
            model_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\trained_models\\logistic_regression\\time_step_57.pkl'
            image_path = 'E:\\group project\\grp_project\\vortex_framework\\data\\\Data\\img57.png'
            vision_model_path = ''
            no_of_bounding_boxes=_main.CVMain(image_path)
            vortex_core = rf_predict_model(dataset_path,model_path,no_of_bounding_boxes)
            print(vortex_core)
else:
    print('Incorrect argument')
