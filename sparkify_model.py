{"metadata":{"language_info":{"name":"python","version":"3.6.6","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import pandas as pd\nimport xgboost as xgb\nfrom sklearn import metrics\nfrom sklearn import preprocessing\nfrom sklearn import ensemble\nfrom sklearn import linear_model\nfrom sklearn.linear_model import LogisticRegression\nimport pandas as pd\nimport xgboost as xgb\nfrom sklearn import metrics\nfrom sklearn import preprocessing\nfrom sklearn import ensemble\nfrom sklearn import linear_model\nfrom sklearn.linear_model import LogisticRegression\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nfrom sklearn import model_selection #\nfrom sklearn import metrics #\n\nimport os\nimport shutil\nimport optuna\n\nimport sklearn.metrics\nimport xgboost as xgb\n\nimport warnings \nwarnings.filterwarnings('ignore')\nimport time\n\nSEED = 108\nN_FOLDS = 5\nCV_RESULT_DIR = \"./xgboost_cv_results\"\nTRAIN_FOLDS=\"/kaggle/working/train_folds.csv\"\nSPARKIFY_USERS=\"/kaggle/input/sparkify-users/Sparkify_users.csv\"\n\ndef create_kfold(df):\n    print(df.columns)\n    print(\"*****Target***\")\n    print(df.Churn.unique())\n    \n    # we create a new column called kfold and fill it with -1\n    df[\"kfold\"] = -1\n     \n    # the next step is to randomize the rows of the data\n    df = df.sample(frac=1).reset_index(drop=True)\n\n    # fetch targets\n    y = df.Churn.values\n    \n    # initiate the kfold class from model_selection module\n    kf = model_selection.StratifiedKFold(n_splits=5)\n\n\n    # fill the new kfold column\n    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):\n        df.loc[v_, 'kfold'] = f\n\n    # save the new csv with kfold column\n    df.to_csv(\"train_folds.csv\", index=False)\n\n\ndef run_logistic(fold):\n    # load the full training data with folds\n    df = pd.read_csv(TRAIN_FOLDS)\n    \n    # all columns are features except id, target and kfold columns\n    features = [\n    f for f in df.columns if f not in (\"Churn\")\n    ]\n    # get training data using folds\n    df_train = df[df.kfold != fold].reset_index(drop=True)\n\n    # ge./t validation data using folds\n    df_valid = df[df.kfold == fold].reset_index(drop=True)\n\n    # get training data\n    x_train = df_train[features].values\n    # get validation data\n    x_valid = df_valid[features].values\n\n    # initialize random forest model\n    model = linear_model.LogisticRegression()\n\n\n    # fit model on training data (ohe)\n    model.fit(x_train, df_train.Churn.values)\n    # predict on validation data\n    # we need the probability values as we are calculating AUC\n    # we will use the probability of 1s\n    valid_preds = model.predict(x_valid)\n    print(valid_preds)\n    # get roc auc score\n    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)\n    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)\n\n    # print auc\n    print(f\"Fold = {fold}, AUC = {auc}\")\n    return auc\n    \n    \ndef run_xgboost(fold):\n    # load the full training data with folds\n    df = pd.read_csv(TRAIN_FOLDS)\n    \n    # all columns are features except id, target and kfold columns\n    features = [\n    f for f in df.columns if f not in (\"Churn\",\"fold\")\n    ]\n    # get training data using folds\n    df_train = df[df.kfold != fold].reset_index(drop=True)\n\n    # ge./t validation data using folds\n    df_valid = df[df.kfold == fold].reset_index(drop=True)\n\n    # get training data\n    x_train = df_train[features].values\n    # get validation data\n    x_valid = df_valid[features].values\n\n    # initialize random forest model\n    model = xgb.XGBClassifier(\n            n_jobs=-1,\n            max_depth=7,\n            n_estimators=200,\n            eval_metric='mlogloss'\n             )\n\n\n    # fit model on training data (ohe)\n    model.fit(x_train, df_train.Churn.values)\n    # predict on validation data\n    # we need the probability values as we are calculating AUC\n    # we will use the probability of 1s\n    valid_preds = model.predict(x_valid)\n    print(valid_preds)\n    # get roc auc score\n    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)\n    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)\n    #roc_auc\n    # print auc\n    print(f\"Fold = {fold},AUC = {auc}\")\n    return auc\n\n\ndef objective(trial):\n    # load the full training data with folds\n    df = pd.read_csv(TRAIN_FOLDS)\n    \n    # all columns are features except id, target and kfold columns\n    features = [\n    f for f in df.columns if f not in (\"Churn\",\"fold\")\n    ]\n    \n    dtrain = xgb.DMatrix(df[features].values, label=df['Churn'])\n\n    param = {\n        \"verbosity\": 1,\n        \"objective\": \"binary:logistic\",\n        \"eval_metric\": \"auc\",\n        \"booster\": trial.suggest_categorical(\"booster\", [\"gbtree\", \"gblinear\", \"dart\"]),\n        \"lambda\": trial.suggest_float(\"lambda\", 1e-8, 1.0, log=True),\n        \"alpha\": trial.suggest_float(\"alpha\", 1e-8, 1.0, log=True),\n        # sampling ratio for training data.\n        \"subsample\": trial.suggest_float(\"subsample\", 0.2, 1.0),\n        # sampling according to each tree.\n        \"colsample_bytree\": trial.suggest_float(\"colsample_bytree\", 0.2, 1.0),\n        \"learning_rate\": trial.suggest_float(\"learning_rate\", 1e-5, 1e-2, log=True)\n    }\n\n    if param[\"booster\"] == \"gbtree\" or param[\"booster\"] == \"dart\":\n        param[\"max_depth\"] = trial.suggest_int(\"max_depth\", 1, 9)\n        # minimum child weight, larger the term more conservative the tree.\n        param[\"min_child_weight\"] = trial.suggest_int(\"min_child_weight\", 2, 10)\n        param[\"eta\"] = trial.suggest_float(\"eta\", 1e-8, 1.0, log=True)\n        param[\"gamma\"] = trial.suggest_float(\"gamma\", 1e-8, 1.0, log=True)\n        param[\"grow_policy\"] = trial.suggest_categorical(\"grow_policy\", [\"depthwise\", \"lossguide\"])\n\n    if param[\"booster\"] == \"dart\":\n        param[\"sample_type\"] = trial.suggest_categorical(\"sample_type\", [\"uniform\", \"weighted\"])\n        param[\"normalize_type\"] = trial.suggest_categorical(\"normalize_type\", [\"tree\", \"forest\"])\n        param[\"rate_drop\"] = trial.suggest_float(\"rate_drop\", 1e-8, 1.0, log=True)\n        param[\"skip_drop\"] = trial.suggest_float(\"skip_drop\", 1e-8, 1.0, log=True)\n\n    xgb_cv_results = xgb.cv(\n        params=param,\n        dtrain=dtrain,\n        num_boost_round=10000,\n        nfold=N_FOLDS,\n        stratified=True,\n        early_stopping_rounds=100,\n        seed=SEED,\n        verbose_eval=False,\n    )\n\n    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().\n    trial.set_user_attr(\"n_estimators\", len(xgb_cv_results))\n\n    # Save cross-validation results.\n    filepath = os.path.join(CV_RESULT_DIR, \"{}.csv\".format(trial.number))\n    xgb_cv_results.to_csv(filepath, index=False)\n\n    # Extract the best score.\n    best_score = xgb_cv_results[\"test-auc-mean\"].values[-1]\n    return best_score\n\ndef run_xgboost_tuned(fold,**best_params):\n    # load the full training data with folds\n    df = pd.read_csv(TRAIN_FOLDS)\n    \n    # all columns are features except id, target and kfold columns\n    features = [\n    f for f in df.columns if f not in (\"Churn\",\"fold\")\n    ]\n    # get training data using folds\n    df_train = df[df.kfold != fold].reset_index(drop=True)\n\n    # get validation data using folds\n    df_valid = df[df.kfold == fold].reset_index(drop=True)\n\n    # get training data\n    x_train = df_train[features].values\n    # get validation data\n    x_valid = df_valid[features].values\n\n    # initialize random forest model\n    model = xgb.XGBClassifier(\n            n_jobs=-1,\n            eval_metric='mlogloss',\n            booster='dart',\n            #lambda = 2.821953496941091e-07,\n            alpha= 2.5646828346432774e-06,\n            subsample= 0.4151045743266873,\n            colsample_bytree= 0.6948040408445149,\n            max_depth= 8,\n            min_child_weight= 2,\n            eta= 0.03515347410218039,\n            gamma= 0.0002775302840172167,\n            grow_policy= 'depthwise',\n            sample_type= 'uniform',\n            normalize_type= 'forest',\n            rate_drop= 1.5923182802351157e-05,\n            skip_drop= 0.01448906401093845,\n            n_estimators= 251,\n            learning_rate=0.00047032141567833225\n              )\n  \n    #model.set_params(**best_params)\n    print(model.get_xgb_params())\n       \n    # fit model on training data (ohe)\n    model.fit(x_train, df_train.Churn.values)\n    # predict on validation data\n    # we need the probability values as we are calculating AUC\n    # we will use the probability of 1s\n    valid_preds = model.predict(x_valid)\n    print(valid_preds)\n    # get roc auc score\n    f1 = metrics.f1_score(df_valid.Churn.values, valid_preds)\n    auc = metrics.roc_auc_score(df_valid.Churn.values, valid_preds)\n    #roc_auc\n    # print auc\n    print(f\"Tuned XGBoost:Fold = {fold},AUC = {auc}\")\n    return auc\n\nif __name__ == \"__main__\":\n    start_time = time.time()\n\n    #create K fold\n    df = pd.read_csv(SPARKIFY_USERS)\n    create_kfold(df)\n    print(\"Kfold completed after --- %s seconds ---\" % (time.time() - start_time))\n\n    #Logistic regression baseline\n    total_auc=0\n    for fold_ in range(5):\n         total_auc+=run_logistic(fold_)\n        \n    mean_auc=total_auc/5\n    print(f\"Mean AUC = {mean_auc}\")\n    print(\"Logistic regression completed after--- %s seconds ---\" % (time.time() - start_time))\n    \n    #XGboost baseline\n    total_auc=0\n    \n    for fold_ in range(5):\n        total_auc+=run_xgboost(fold_)\n\n    mean_auc=total_auc/5\n    print(f\"Baseline XGBoost Mean AUC = {mean_auc}\")\n    \n    print(\"Baseline XGBoost model after--- %s seconds ---\" % (time.time() - start_time))\n    \n    #Optuna hyperparam study\n    if not os.path.exists(CV_RESULT_DIR):\n        os.mkdir(CV_RESULT_DIR)\n\n    study = optuna.create_study(direction=\"maximize\")\n    study.optimize(objective, n_trials=100, timeout=600)\n\n    print(\"Number of finished trials: \", len(study.trials))\n    print(\"Best trial:\")\n    trial = study.best_trial\n    best_params = study.best_params\n\n    print(\"  Value: {}\".format(trial.value))\n    print(\"  Params: \")\n    for key, value in trial.params.items():\n        print(\"    {}: {}\".format(key, value))\n\n    print(\"  Number of estimators: {}\".format(trial.user_attrs[\"n_estimators\"]))\n    print(\"Best Params: {}\".format(best_params))\n    shutil.rmtree(CV_RESULT_DIR)    \n    \n    print(\"Parameter tuning completed after--- %s seconds ---\" % (time.time() - start_time))\n    \n    #Tuned XGboost\n    total_auc=0\n    for fold_ in range(5):\n        total_auc+=run_xgboost_tuned(fold_)#,**best_params)\n\n    mean_auc=total_auc/5\n    print(f\"Tuned XGBooost Mean AUC = {mean_auc}\")\n    print(\"XGBoost Hyperparameter tuned complete after--- %s seconds ---\" % (time.time() - start_time))\n    \n    \n        \n        ","metadata":{"collapsed":false,"_kg_hide-input":false,"jupyter":{"outputs_hidden":false}},"execution_count":null,"outputs":[]}]}