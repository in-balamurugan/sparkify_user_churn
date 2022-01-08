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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection  #
from sklearn import metrics  #
from sklearn.ensemble import RandomForestClassifier
import os
import shutil
import optuna

import sklearn.metrics
import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')
import time
import pickle
import shap
from matplotlib import pyplot as plt

MODEL_NAME = "xgb_full.pkl"
SEED = 108
N_FOLDS = 5
CV_RESULT_DIR = "./xgboost_cv_results"
TRAIN_FOLDS = "output/train_folds.csv"
SPARKIFY_USERS = "data/Sparkify_users.csv"
tuned_params = {'objective': 'binary:logistic', 'base_score': 0.5, 'booster': 'gbtree',
                'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.8702730765483531,
                'gamma': 0.6041310184357518, 'gpu_id': -1, 'interaction_constraints': '', 'learning_rate': 0.300000012,
                'max_delta_step': 0, 'max_depth': 7, 'min_child_weight': 4, 'monotone_constraints': '()',
                'n_jobs': -1, 'num_parallel_tree': 1, 'predictor': 'auto', 'random_state': 0, 'reg_alpha': 0,
                'reg_lambda': 1, 'scale_pos_weight': 1,
                'subsample': 0.799542263147059, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None,
                'eval_metric': 'auc'}


def create_kfold(df):
    """creates and writes a stratified 5-folds on the input dataset

    Args:
      df: the input dataframe

    Returns:
      None

    """
    print(df.dtypes)

    # df.drop(['userId'], axis=1, inplace=True)
    print(df.dtypes)
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
    """creates and writes a stratified 5-folds on the input dataset
       Args:
          df: the input dataframe
        Returns:
          None
        """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # df.drop(['userId'], axis=1, inplace=True)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
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
    model = linear_model.LogisticRegression(solver='liblinear')

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


def run_randomforest(fold):
    """Trains data on random forest and tests on validation data  passed in fold
    Args:
      fold(int): indicating the number of fold between 0 and 4
    Returns:
      AUC(float):AUC of the fold
    """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # df.drop(['userId'], axis=1, inplace=True)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
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
    model = RandomForestClassifier()

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
    """Trains data on untuned XGBoost and tests on validation data  passed in fold
        Args:
          fold(int): indicating the number of fold between 0 and 4
        Returns:
          AUC(float):AUC of the fold
        """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
    ]
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values

    # initialize XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=1,
        n_estimators=10,
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
    # roc_auc
    # print auc
    print(f"Fold = {fold},AUC = {auc}")
    return auc


def objective(trial):
    """Performs study on full dataset to tune the best hyperparameters and returns best score object
        Args:
          trial(Object): Optuna object
        Returns:
          best_scoreAUC(Object): Object of the best performing hyperparameter set
        """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
    ]

    dtrain = xgb.DMatrix(df[features].values, label=df['Churn'])

    param = {"verbosity": 0, "objective": "binary:logistic", "eval_metric": "auc",
             "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
             "subsample": trial.suggest_float("subsample", 0.2, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
             "max_depth": trial.suggest_int("max_depth", 1, 9),
             "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
             "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
             "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)}

    # minimum child weight, larger the term more conservative the tree.

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


def run_xgboost_tuned(fold):
    """Trains data on tuned params and tests on validation data  passed in fold
        Args:
          fold(int): indicating the number of fold between 0 to 4
        Returns:
          AUC(float):AUC of the fold
        """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
    ]
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values

    # initialize XGB model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        eval_metric='auc',
        n_estimators=383
    )

    model.set_params(**tuned_params)

    # model.set_params(**best_params)
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
    # roc_auc
    # print auc
    print(f"Tuned XGBoost:Fold = {fold},AUC = {auc}")
    return auc


def print_important_features():
    """Final model trained on full set and important features are extracted through Shap model explainer function
        Args: None
        Returns: None
        """
    # load the full training data with folds
    df = pd.read_csv(TRAIN_FOLDS)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("Churn", "fold", "userId", "obsDays")
    ]

    df_train = df.reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # initialize XGB model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        eval_metric='auc',
        n_estimators=383
    )

    model.set_params(**tuned_params)

    # fit model on training data (ohe)
    model.fit(x_train, df_train.Churn.values)

    explainer = shap.Explainer(model)
    shap_values = explainer(df_train[features])

    # summarize the effects of all the features
    shap.plots.waterfall(shap_values[0])
    plt.savefig('shap_waterfall.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    shap.plots.beeswarm(shap_values)
    plt.savefig('shap_beeswarm.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    shap.plots.bar(shap_values)
    plt.savefig('shap_bar.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()

    # save model
    pickle.dump(model, open(MODEL_NAME, "wb"))


if __name__ == "__main__":
    start_time = time.time()

    # create K fold
    df = pd.read_csv(SPARKIFY_USERS)
    create_kfold(df)
    print("Kfold completed after --- %s seconds ---" % (time.time() - start_time))

    # Logistic regression
    total_auc = 0
    for fold_ in range(5):
        total_auc += run_logistic(fold_)

    mean_auc = total_auc / 5
    print(f"Mean AUC = {mean_auc}")
    print("Logistic regression completed after--- %s seconds ---" % (time.time() - start_time))

    # Random forest classifier
    total_auc = 0
    for fold_ in range(5):
        total_auc += run_randomforest(fold_)

    mean_auc = total_auc / 5
    print(f"Random forest Mean AUC = {mean_auc}")
    print("Random forest completed after--- %s seconds ---" % (time.time() - start_time))

    # XGBoost classifier
    total_auc = 0
    for fold_ in range(5):
        total_auc += run_xgboost(fold_)

    mean_auc = total_auc / 5
    print(f"XGBoost Mean AUC = {mean_auc}")
    print("XGBoost completed after--- %s seconds ---" % (time.time() - start_time))

    # Optuna hyperparam study
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

    # Tuned XGboost
    total_auc = 0
    for fold_ in range(5):
        total_auc += run_xgboost_tuned(fold_)  # ,**best_params)

    mean_auc = total_auc / 5
    print(f"Tuned XGBooost Mean AUC = {mean_auc}")
    print("XGBoost Hyperparameter tuned complete after--- %s seconds ---" % (time.time() - start_time))

    print_important_features()
    print("Feature importance complete after--- %s seconds ---" % (time.time() - start_time))
