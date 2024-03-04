# End To End ML Project: Credit Card Default Prediction

### Problem Statement : Credit Card Default Prediction

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

**Python 3.9**

**Flask**

**MongoDB**

**Numpy**

**Pandas**

**Scikit-Learn**

**Matplotlib**

**Seaborn**

**Xgboost**


## Run Locally

### Created an environment

```
conda create -p env python=3.9
```
### Activate the environment

```
source activate ./env
```

### Clone the Project

```bash
  git clone https://github.com/SuchindraKumar/Credit-Card-Default-Prediction.git
```

### Go to the Project Directory

```bash
  cd Credit-Card-Default-Prediction
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







## Screenshots

![App Screenshot]([https://via.placeholder.com/468x300?text=App+Screenshot+Here](https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/Home_Page.png)https://github.com/SuchindraKumar/Credit-Card-Default-Prediction/blob/main/images/Home_Page.png)



## Authors

- [@SuchindraKumar](https://github.com/SuchindraKumar)
