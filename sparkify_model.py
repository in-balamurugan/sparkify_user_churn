import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection #
from sklearn import metrics #

import os
import shutil
import optuna

import sklearn.metrics
import xgboost as xgb

import warnings 
warnings.filterwarnings('ignore')
import time

SEED = 108
N_FOLDS = 5
CV_RESULT_DIR = "./xgboost_cv_results"
TRAIN_FOLDS="/kaggle/working/train_folds.csv"
SPARKIFY_USERS="/kaggle/input/sparkify-users/Sparkify_users.csv"

def create_kfold(df):
    print(df.columns)
    print("*****Target***")
    print(df.Churn.unique())
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
     
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df.Churn.values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)


    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)


def run_logistic(fold):
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)
    
    # all columns are features except id, target and kfold columns
    features = [
    f for f in df.columns if f not in ("Churn")
    ]
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # ge./t validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values

    # initialize random forest model
    model = linear_model.LogisticRegression()


    # fit model on training data (ohe)
    model.fit(x_train, df_train.Churn.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict(x_valid)
    print(valid_preds)
    # get roc auc score
    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)
    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
    return auc
    
    
def run_xgboost(fold):
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)
    
    # all columns are features except id, target and kfold columns
    features = [
    f for f in df.columns if f not in ("Churn","fold")
    ]
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # ge./t validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values

    # initialize random forest model
    model = xgb.XGBClassifier(
            n_jobs=-1,
            max_depth=7,
            n_estimators=200,
            eval_metric='mlogloss'
             )


    # fit model on training data (ohe)
    model.fit(x_train, df_train.Churn.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict(x_valid)
    print(valid_preds)
    # get roc auc score
    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)
    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)
    #roc_auc
    # print auc
    print(f"Fold = {fold},AUC = {auc}")
    return auc


def objective(trial):
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)
    
    # all columns are features except id, target and kfold columns
    features = [
    f for f in df.columns if f not in ("Churn","fold")
    ]
    
    dtrain = xgb.DMatrix(df[features].values, label=df['Churn'])

    param = {
        "verbosity": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=10000,
        nfold=N_FOLDS,
        stratified=True,
        early_stopping_rounds=100,
        seed=SEED,
        verbose_eval=False,
    )

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(xgb_cv_results))

    # Save cross-validation results.
    filepath = os.path.join(CV_RESULT_DIR, "{}.csv".format(trial.number))
    xgb_cv_results.to_csv(filepath, index=False)

    # Extract the best score.
    best_score = xgb_cv_results["test-auc-mean"].values[-1]
    return best_score

def run_xgboost_tuned(fold,**best_params):
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)
    
    # all columns are features except id, target and kfold columns
    features = [
    f for f in df.columns if f not in ("Churn","fold")
    ]
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values

    # initialize random forest model
    model = xgb.XGBClassifier(
            n_jobs=-1,
            eval_metric='mlogloss',
            booster='dart',
            #lambda = 2.821953496941091e-07,
            alpha= 2.5646828346432774e-06,
            subsample= 0.4151045743266873,
            colsample_bytree= 0.6948040408445149,
            max_depth= 8,
            min_child_weight= 2,
            eta= 0.03515347410218039,
            gamma= 0.0002775302840172167,
            grow_policy= 'depthwise',
            sample_type= 'uniform',
            normalize_type= 'forest',
            rate_drop= 1.5923182802351157e-05,
            skip_drop= 0.01448906401093845,
            n_estimators= 251,
            learning_rate=0.00047032141567833225
              )
  
    #model.set_params(**best_params)
    print(model.get_xgb_params())
       
    # fit model on training data (ohe)
    model.fit(x_train, df_train.Churn.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict(x_valid)
    print(valid_preds)
    # get roc auc score
    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)
    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)
    #roc_auc
    # print auc
    print(f"Tuned XGBoost:Fold = {fold},AUC = {auc}")
    return auc

if __name__ == "__main__":
    start_time = time.time()

    #create K fold
    df = pd.read_csv(SPARKIFY_USERS)
    create_kfold(df)
    print("Kfold completed after --- %s seconds ---" % (time.time() - start_time))

    #Logistic regression baseline
    total_auc=0
    for fold_ in range(5):
         total_auc+=run_logistic(fold_)
        
    mean_auc=total_auc/5
    print(f"Mean AUC = {mean_auc}")
    print("Logistic regression completed after--- %s seconds ---" % (time.time() - start_time))
    
    #XGboost baseline
    total_auc=0
    
    for fold_ in range(5):
        total_auc+=run_xgboost(fold_)

    mean_auc=total_auc/5
    print(f"Baseline XGBoost Mean AUC = {mean_auc}")
    
    print("Baseline XGBoost model after--- %s seconds ---" % (time.time() - start_time))
    
    #Optuna hyperparam study
    if not os.path.exists(CV_RESULT_DIR):
        os.mkdir(CV_RESULT_DIR)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    best_params = study.best_params

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Number of estimators: {}".format(trial.user_attrs["n_estimators"]))
    print("Best Params: {}".format(best_params))
    shutil.rmtree(CV_RESULT_DIR)    
    
    print("Parameter tuning completed after--- %s seconds ---" % (time.time() - start_time))
    
    #Tuned XGboost
    total_auc=0
    for fold_ in range(5):
        total_auc+=run_xgboost_tuned(fold_)#,**best_params)

    mean_auc=total_auc/5
    print(f"Tuned XGBooost Mean AUC = {mean_auc}")
    print("XGBoost Hyperparameter tuned complete after--- %s seconds ---" % (time.time() - start_time))
    
    
        
        