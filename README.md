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

# 3. Data preparation for training the classic Machine Learning models

# 4. Regression problem

** Problem description** 

## 4.1 Baseline 

Launch
```
python3 models/regression/baseline.py
```

## 4.2 XGBoost

Launch
```
python3 models/regression/xgboost.py
```

Folder `results/`
