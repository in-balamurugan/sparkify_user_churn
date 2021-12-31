# Sparkify User churn prediction Project
A model to predict the churn of users from the music platform Sparkify. Blog details here.


## Table of Contents

1. [Folders](#Folders)
2. [Dependecies](#Dependecies)
3. [Instructions](#Instructions)
4. [Metrics](#Metrics)
5. [Credits](#Credits)


<a name="Folders"></a>
### Folders:
|--data<br>
|    &nbsp; &nbsp;&nbsp;  -- sparkify_users.csv # data to process<br>
|--models<br>
|    &nbsp; &nbsp;&nbsp;  -- SparkifyPreProcessing.ipynb #Processing in Spark<br>
|    &nbsp; &nbsp;&nbsp;  -- Sparkify_model.py  #Building ML model<br>
--&nbsp; README.md<br>


<a name="Dependecies"></a>
### Package Dependecies:
- os
- pandas
- pyspark
- pandas
- sklearn
- shutil
- optuna
- xgboost

<a name="Instructions"></a>
### Instructions:
1. Creating EMR Instance with below configuration to process the input file "s3n://udacity-dsnd/sparkify/sparkify_event_data.json" to process the data with Spark

	> m5.xlarge<br>
    > vCore, 16 GiB memory, EBS onlystorage<br>
   > EBS Storage:32 GiB<br>

2. Run SparkifyPreProcessing.ipynb to produce sparkify_user.csv

3. Run sparkify_model.py to produce the final model and metrics
 
   
<a name="Metrics"></a>
### Metrics:
* Execution time: 13 minutes for model building
* Logistic Regression Mean AUC = 0.8182947530020286
* Baseline XGBoost Mean AUC = 0.9875935902612601
* Tuned XGBooost Mean AUC = 0.9907856525179539

<a name="Credit"></a>
### Credit:
* Udacity
* Book: Approaching Almost Any Machine learning porblem
* Hyperparameter tuning:Optuna