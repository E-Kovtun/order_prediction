Tables with actual results: https://docs.google.com/spreadsheets/d/1pNlDeTu0y6qOa4XepUx7wDxxQo-HiY_8ywSdLeCqohc/edit?usp=sharing

# 1. Data overview and preprocessing 

The dataset has the following structure:

![Dataset_Example](/images/dataset_example.png)

#### Meanings of the columns:
* **Ship.to** - restaurant id
* **PLZ** - postal code of the restaurant
* **Material** - material id that was ordered 
* **Delivery_Date_week** - date of the order delivery
* **Status** - status of the restaurant
* **MaterialGroup.1**, **MaterialGroup.2**, **MaterialGroup.3** - characteristics of the material
* **Amount_HL** - amount of ordered material

#### Steps of data preprocessing:

1. Add two columns with day and month of order (**Day**, **Month**)
2. Ordinal encoding of categorical features (**Ship.to**, **PLZ**, **Material**, **Day**, **Month**, **Status**, **MaterialGroup.1**, **MaterialGroup.2**, **MaterialGroup.4**)
3. There are cases when several different materials are ordered at the same day by one restaurant. For example:

![Rows_Combination](/images/rows_combination.png)

Such rows are combined into a single one by summing up **Amount_HL** and joining the values of **Material**, **MaterialGroup.1**, **MaterialGroup.2** and **MaterialGroup.4** features

4. Add column **dt** which is responsible for time difference (in days) between consecutive orders for each restaraunt 

The resulting dataset has the following form:

![Preprocessed_Dataset](/images/preprocessed_dataset.png)

# 2. Train-test split

Train data contains history of orders from 2012-12-30 till 2017-12-31 (periods with available information are different for each particular restaurant). Test data provides infromation on orders from 2018-01-07 till 2018-07-15 for each restaurant that is present in the train data.

# 3. Feature processing for training/testing the classic Machine Learning models

In order to train and test the classic Machine Learning models, we need to process the features in an appropriate manner. 

* Features **Ship.to**, **PLZ**, **Day**, **Month**, **Status** are encoded using a one-hot encoding scheme
* Features **Material**, **MaterialGroup.1**, **MaterialGroup.2**, **MaterialGroup.4** are turned into vectors with Bag of Words procedure
* Features **dt**, **Amount_HL**, **Material**, **MaterialGroup.1**, **MaterialGroup.2**, **MaterialGroup.4** are scaled with Standard Scaler

The main task is to predict some information in a particular day using information from several prior timestamps. The parameter **look_back** is responsible for the number of timestamps which can be used to construct the prediction for the following timestamp. If **look_back** is equal to 4, then we take 4 rows that relate to one restaurant and concatenate them into one row (features **Ship.to** and **PLZ** are appeared only one time). The target variable might be some value from the next row (that follows these 4 considered rows). 

# 4. Regression problem

In the regression problem we are interested in prediction of **Amount_HL**. So, for each restaurant we use all information (including **Amount_HL**) from several subsequent timestamps and predict the value of **Amount_HL** for the next timestamp. 

## 4.1 Baseline 

In the baseline model we actually need only the test set and feature processing step can be skipped. To predict the **Amount_HL** value for a particular timestamp, we average previous values of **Amount_HL** (the number of values that we average are specified by **look_back** parameter).

To launch the baseline model:

```
python3 models/regression/baseline.py
```

## 4.2 XGBoost

For this model all features are processed by the way described above. 

To launch the XGBoost model:
```
python3 models/regression/xgboost.py
```
r2_score for the test set will be saved in the `results/` folder
