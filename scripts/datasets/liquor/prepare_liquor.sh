export PYTHONPATH=.
INIT_DATA_FOLDER="../data/initial/liquor/"
mkdir -p ${INIT_DATA_FOLDER}
cd ${INIT_DATA_FOLDER}
echo "Downloading data..."
gdown --fuzzy https://drive.google.com/file/d/1kFvjkQzb_Pc41B50WcE3M8KJG0d-s4Ho/view?usp=sharing
cd -
PREPARED_DATA_FOLDER="../data/prepared/liquor/"
mkdir -p ${PREPARED_DATA_FOLDER}
echo "Splitting datasets into train-valid-test..."
python3 scripts/datasets/liquor/handle_liquor.py ${INIT_DATA_FOLDER} ${PREPARED_DATA_FOLDER}
