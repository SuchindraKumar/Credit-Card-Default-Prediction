# End To End ML Project: Credit Card Default Prediction

(https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/demo/ccfd.png)

### Problem Statement :

The Task at hand is to create a Predictive Model that can Precisely Forecast if a Credit Card Holder will experience a Payment Default in the near future. Financial institutions need to be able to predict Credit Card Defaults in order to evaluate the risk of granting credit and make well-informed lending decisions

# Dataset Description:
The "Default of Credit Card Clients" dataset provides valuable insights into credit card default behavior among clients in Taiwan. With 25 variables covering demographic factors, credit data, payment history, and bill statements, the dataset offers a comprehensive view of the factors influencing default payments. Here's a breakdown of potential exploration and analysis:

1. **Demographic Analysis**: Explore how default payment probability varies across different demographic variables such as gender, education, marital status, and age. This analysis could reveal if certain demographic groups are more prone to default than others.

2. **Repayment Behavior Analysis**: Investigate the relationship between repayment behavior (PAY_0 to PAY_6) and default payments. Determine if clients with a history of payment delays or defaults are more likely to default in the following months.

3. **Bill Statement Analysis**: Examine the relationship between bill statement amounts (BILL_AMT1 to BILL_AMT6) and default payments. Analyze if clients with higher bill amounts are more likely to default, or if there's a specific trend in bill amounts preceding default.

4. **Previous Payment Analysis**: Explore the relationship between previous payment amounts (PAY_AMT1 to PAY_AMT6) and default payments. Determine if clients who made larger or smaller payments in previous months are more likely to default.

5. **Predictive Modeling**: Build predictive models to identify the strongest predictors of default payment. Utilize techniques such as logistic regression, decision trees, or random forests to identify which variables have the most significant impact on predicting default.

6. **Feature Importance Analysis**: Conduct feature importance analysis to rank variables based on their predictive power for default payment. This can help prioritize variables for further investigation or feature selection in predictive modeling.

7. **Visualization**: Use visualizations such as histograms, box plots, and correlation matrices to gain insights into the distribution of variables and their relationships with default payments.

By exploring these avenues of analysis, we can gain a deeper understanding of the factors influencing credit card default behavior and identify strategies for mitigating default risk. This knowledge can inform credit risk management practices and aid in decision-making processes for lenders and financial institutions.


## Tech Stack

**Python 3.9** LINK: https://www.python.org/downloads/

**Flask**

**MongoDB :** 

**MongoDB Community Server** LINK: https://www.mongodb.com/try/download/community-kubernetes-operator

 **MongoDB Compass** LINK: https://www.mongodb.com/try/download/compass

**Numpy**

**Pandas**

**Scikit-Learn**

**Matplotlib**

**Seaborn**

**Xgboost**


## Run Locally

### Clone the Project

```bash
  git clone https://github.com/SuchindraKumar/Credit-Card-Default-Prediction.git
```


### Created an environment

```
conda create -p env python=3.9
```
### Activate the environment

```
source activate ./env
```


### Install dependencies

```
pip install -r requirements.txt
```
### Train The Model
```
python src/pipeline/training_pipeline.py
```

### Run

```
python app.py
```
### Now,
```bash
Open Up Your local-host and Port Number 8080
```


## Deployment

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	# With Specific Access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	# Description: About the deployment

	1. Build Docker Image of the Source Code

	2. Push your Docker Image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	# Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/text-s

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	# Optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	# Required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app



## Screenshots

![App Screenshot](https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/Home_Page.png)


![App Screenshot](https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/Prediction_Form_1.png)
![App Screenshot](https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/Prediction_Form_2.png)


![App Screenshot](https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/prediction_result.png)



## Authors

- [@SuchindraKumar](https://github.com/SuchindraKumar)
