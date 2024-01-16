Tables with actual results: https://docs.google.com/spreadsheets/d/1pNlDeTu0y6qOa4XepUx7wDxxQo-HiY_8ywSdLeCqohc/edit?usp=sharing

# 1. Managing different datasets 

Scripts for downloading and preparing diffrent datasets are located in `scripts/datasets/`. For example, command for getting "liquor" dataset:

``` bash scripts/datasets/liquor/prepare_liquor.sh ```

The exception is "order" dataset, for which you need place initial files in corresponding folder mannually. 

# 2. Model training and evaluation

All models parametrs are specified in `configs/base.json` and `configs/train_params.json`.

To launch model training and testing procedure:

``` bash scripts/run.sh```


