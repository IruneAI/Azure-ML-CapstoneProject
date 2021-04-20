
# Heart Disease Analysis & Prediction

Heart disease is the leading cause of death for men, women, and people of most racial and ethnic groups in the United States. **One person dies every 36 seconds in the United States from cardiovascular disease. About 655,000 Americans die from heart disease each year—that's 1 in every 4 deaths**. 


![deaths_map](/starter_file/images/fs_heart_disease.png)

![deaths](/starter_file/images/percentages_deaths_stratified_by_ethnic_gender.png)
**Fig 1. Deaths stratified by race, ethnic and gender. From Centers for Disease Control and Prevention [(CDC)](https://www.cdc.gov/)**

In this project, I will be approaching a **classification problem** for Heart Disease *understanding* as well as predict the *presence of heart disease in a patient*. For that, I will be working under [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/) context and exploiting different capabilities for supporting the **end to end ML workflow** on [Azure](https://azure.microsoft.com/en-us/) CSP. 

## Dataset

### Overview

The dataset used by this project was available under [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci). This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient (0: no presence, 1: presence)

![data_accesibility](/starter_file/images/data_accesibility.png)
**Fig 2. External Data Accesibility**


### Task

The task will be a binay classication task to 'determine'the presence of a heart disease (being target [0,1]). For that, the following features will be used:
- age
- sex
- chest pain type (4 values)
- resting blood pressure
- serum cholestoral in mg/dl
- fasting blood sugar > 120 mg/dl
- resting electrocardiographic results (values 0,1,2)
- maximum heart rate achieved
- exercise induced angina
- oldpeak = ST depression induced by exercise relative to rest
- the slope of the peak exercise ST segment
- number of major vessels (0-3) colored by flourosopy
- thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

### Access
The dataset will be accesible/exposed within AzureML datastore.

## Automated ML
The concept of utomated ML isthe process of automating the tasks of applying ML to real-world problems. Usually it helps to automate the whole ML workflow/pipeline enabling the data scientist or even non-experts to leverage ML in a efficient manner. 

In this case, Azure AutoML gave me the possibility to try multiple algorithms in this classification task. Due to the high balance of the data "primary_metric" : 'accuracy' was chosen. As compute target a STANDARD_D2_V2 was used with min_nodes=0 and max_nodes=4.

automl_settings = {
    "name": "AutoML_Demo_Experiment_{0}".format(time.time()),
    "experiment_timeout_minutes" : 30,
    "enable_early_stopping" : True,
    "iteration_timeout_minutes": 10,
    "n_cross_validations": 5,
    "primary_metric": 'accuracy',
    "max_concurrent_iterations": 10,
}

automl_config = AutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             compute_target=compute_target,
                             training_data=dataset,
                             label_column_name='target',
                             **automl_settings,
                             )


### Results


*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?


![run_details_automl](/starter_file/images/run_details_automl.png.png)
**Fig 3. Run Details **


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
Please follow the following link: https://drive.google.com/file/d/1SfNEbAA6gM5WbYy-Nse2F8Hrmr8imoi7/view?usp=sharing

## Standout Suggestions
Additional manual feature engineering as well as ONNX support. Also, new datasets integration as well as correlated ones like Stroke. 
