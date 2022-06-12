export PYTHONPATH=.
INIT_DATA_FOLDER="../data/initial/gender/"
mkdir -p ${INIT_DATA_FOLDER}
cd ${INIT_DATA_FOLDER}
echo "Downloading data..."
gdown --fuzzy https://drive.google.com/file/d/1GJVT1Hq_gxwBzTb1JtOiXQow5tbtoTqe/view?usp=sharing
cd -
PREPARED_DATA_FOLDER="../data/prepared/gender/"
mkdir -p ${PREPARED_DATA_FOLDER}
echo "Splitting datasets into train-valid-test..."
python3 scripts/datasets/gender/handle_gender.py ${INIT_DATA_FOLDER} ${PREPARED_DATA_FOLDER}
