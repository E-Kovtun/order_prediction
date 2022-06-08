export PYTHONPATH=.
INIT_DATA_FOLDER="../data/initial/sales/"
mkdir -p ${INIT_DATA_FOLDER}
cd ${INIT_DATA_FOLDER}
echo "Downloading data..."
gdown --fuzzy https://drive.google.com/file/d/1y3X8HTaSmv4PuJ4LnAlxLDsr3kHWhBn8/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1apLICvwOAXLvGh8sYJdivb7Iipw5cntZ/view?usp=sharing
cd -
PREPARED_DATA_FOLDER="../data/prepared/sales/"
mkdir -p ${PREPARED_DATA_FOLDER}
echo "Splitting datasets into train-valid-test..."
python3 /NOTEBOOK/UHH/Repository/Repository_LSTM/Repository/scripts/datasets/sales/handle_sales.py ${INIT_DATA_FOLDER} ${PREPARED_DATA_FOLDER}
