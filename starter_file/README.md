
# Heart Disease Analysis & Prediction

Heart disease is the leading cause of death for men, women, and people of most racial and ethnic groups in the United States. **One person dies every 36 seconds in the United States from cardiovascular disease. About 655,000 Americans die from heart disease each year—that's 1 in every 4 deaths**. 


![deaths_map](/starter_file/images/fs_heart_disease.png)

![deaths](/starter_file/images/percentages_deaths_stratified_by_ethnic_gender.png)
**Fig 1. Deaths stratified by race, ethnic and gender. From Centers for Disease Control and Prevention [(CDC)](https://www.cdc.gov/)**

In this project, I will be approaching a **classification problem** for Heart Disease *understanding* as well as predict the *presence of heart disease in a patient*. For that, I will be working under [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/) context and exploiting different capabilities for supporting the **end to end ML workflow** on [Azure](https://azure.microsoft.com/en-us/) CSP. 

The most interesting part of this project is that it shows the great power of **AutoML** as well as **Hyperdrive** under Azure. AutoML will provide us the possibility to create a benchmarking of different algorithms getting multiple models really **fast** and in an **efficient** way. Hyperdrive, howewer, will help us to boost the hyper parameters tuning/optimization with multiple approaches available for exploring and exploiting the parameters search space. Concretely, in this project, I decided to go from *simplest* classifcation approach (Logistic Regression) for hyperparameters tuning identifying best possible model and going further benchmkarkign different algorithms thanks to Azure AutoML. 

Finally, thanks to combination of these approaches, **best model** was selected for further deployment and consumption. See model deployment section for further details.

## Dataset

### Overview

The dataset used by this project was available under [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci). This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient (0: no presence, 1: presence)

Since dataset is outside the Azure environment, this should have been registered. For that different approaches could be carrying out. In this case, the dataset was registered within AzureML Studio and uploaded to an allocated Azure Blob Storage (datastore) for further analysis and train/test. The data has been made available and consumed programmatically under current created Azure workspace (in this case workspaceblobstore).

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
The concept of automated ML ist he process of automating the tasks of applying ML to real-world problems. Usually it helps to optimize the whole ML workflow/pipeline enabling the data scientist or even non-experts to leverage ML in a efficient manner. 

In this case, Azure AutoML give us the possibility to try multiple algorithms (in this case for solving a classification task). Due to nature of the data (pretty balanced) accurary was selected as "primary_metric". As compute target a STANDARD_D2_V2 was used with min_nodes=0 and max_nodes=4.

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



![run_details_automl](/starter_file/images/run_details_automl.png)

![run_details_automl](/starter_file/images/run_details_automl2.png)
**Fig 3. Run Details**


![best_model_automl](/starter_file/images/best_model_automl.png)
![best_model_automl](/starter_file/images/best_model_autml.png)

**Fig 4. Best model Automl**


## Hyperparameter Tuning
The main objective of usign Hyperdrive is to be able to find the configuration of hyperparameters that results in the best performance. This search space is usually computationally expensive as well as manual, so having an automated and efficient way to do is always interesting to explore.

Different parameters could be optimizing depending of the algorithm we are working on. In this case, I decided to go to explore the parameter search space for the first simple approach (Occar razor): Logistic Regression. 

One key aspect of the hyperparameters tuning is the actual definition of the *search space as well as *distributions* they are going to follow and *sampling methods*. This would impact greatly the performance of the models. Also, a primary metric definition is key for letting the exploration and explotation optimization process the direction it should follow based on the metric to be optimized (accuracy in this case). 

The first approach selected for sampling was 'random sampling' over Grid and Bayesian. From Microsoft [documentation] (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) it supports both discrete and continuous hyperparameters as well as another important concept: **early termination** of low-performance runs, which help us saving computational resources. This is a good first approach to carry out and then refine the search space to improve results.

Regading the early termination policy MedianStoppingPolicy was used. The interesting point is that this policy 'stops runs whose primary metric value is worse than the median of the averages'. 
Hyperparameters explored were '--C' Inverse of regularization strength and '--max_iter' Maximum number of iterations to converge. These parameters optimization is pretty important not only for enabling convergence, but also to avoid eoverfitting as is the case of C parameter.

Please refer to the code for further information.

### Results
![hyperparameters_details](/starter_file/images/hyperparameters_details.png)
**Fig 5. Hyperparameters Details**

![hyperparameters_details](/starter_file/images/hyper_completed.png)
**Fig 6. Hyperparameters Completed**

## Model Deployment

Best model (best performance run on Accuracy) identified from AutoML was deployed. Deployment was done under the following configurations:

- Number of CPU cores to allocate for this Webservice: cpu_cores=1,
- memory_gb=1 (selft explanatory)
- description='predicting heart diseases',
- auth_enabled=True (Important feature for a secure consumption, see the screenshot below.
- enable_app_insights= True,
- collect_model_data = True

Please see below the healthy status of the service as well as the testing HTTP requests with a dummy payload.


![healthy_deployed_service](/starter_file/images/healthy_deployed_service.png)
**Fig 07. Healthy status for deployed service: heart-disease-service**


![model_testing_endpoint](/starter_file/images/model_testing_endpoint.png)
**Fig 8. Testing endpoint programatically.**



## Screen Recording
Please follow the following link: https://drive.google.com/file/d/1SfNEbAA6gM5WbYy-Nse2F8Hrmr8imoi7/view?usp=sharing

## Future Improvements
Additional manual feature engineering as well as ONNX support. Also, new datasets integration as well as correlated ones like Stroke. 
Explore additional parameters under Hyperdrive such as other sampling methods or add new parameters to the searching process. 
