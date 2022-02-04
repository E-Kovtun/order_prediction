Tables with actual results: https://docs.google.com/spreadsheets/d/1pNlDeTu0y6qOa4XepUx7wDxxQo-HiY_8ywSdLeCqohc/edit?usp=sharing

# I. Restaurants Orders Dataset

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

1. We leave only the features which relate to ids: **Ship.to**, categories: **Material**, amounts: **Amount_HL**, dates: **Delivery_Date_week** in order to build general models that can be used for other datasaets with a similar structure.
2. Ordinal encoding of categorical features (**Ship.to** and **Material**).
3. There are cases when several different materials are ordered at the same day by one restaurant. For example:

![Rows_Combination](/images/rows_combination.png)

Such rows are combined into a single one by joining the values of **Material** and **Amount_HL**.

4. Add column **dt** which is responsible for time difference (in days) between consecutive orders for each restaraunt.
5. Features **Amount_HL** and **dt** are scaled with MinMaxScaler, taking the minimum amd maximum among all values encoutered in the dataset.

The resulting dataset has the following form:

![Preprocessed_Dataset](/images/preprocessed_dataset_upd.png)

# 2. Train-valid-test split

Train data contains history of orders from 2012-12-30 till 2017-12-31 (periods with available information are different for each particular restaurant). Test data provides infromation on orders from 2018-01-07 till 2018-07-15 for each restaurant that is present in the train data. Fot the training of neural networks we need validation dataset. We construct it from the initial train data. For each restaurant id we take 80% of all unique days for which we know the order history for the new train dataset and the remaining 20% for the validation dataset. 

# 3. Regression Problem. Prediction of Amounts.  

In this problem we are interested in prediction of **Amount_HL**. So, for each restaurant we use all information (including **Amount_HL**) from previous subsequent timestamps and predict the vector of **Amount_HL** (each value of the vector corresponds to the amount for a particular material id which is known) for the current timestamp. 

Statistics of **Amount_HL** variable (for all data):

| Statistic   | **Amount_HL** |
| ----------- | ----------- |
| mean        | 0.704       |
| std         | 1.026       |
| quantile 0.    | 0.06        |
| quantile 0.25  | 0.15        |
| quantile 0.50  | 0.4         |
| quantile 0.75  | 1.0         |
| quantile 1.    | 110.0       |

Metrics for evaluation regression:
* R2_score
* MAPE


#### 3.1 Prediction of total amount 
#### 3.1 Prediction of vector of amounts 




