export PYTHONPATH=.
INIT_DATA_FOLDER="../data/initial/demand/"
mkdir -p ${INIT_DATA_FOLDER}
cd ${INIT_DATA_FOLDER}
echo "Downloading data..."
gdown --fuzzy https://drive.google.com/file/d/1nui-yK394gpDl-ViLTF7NEDi5k2nQQYL/view?usp=sharing
cd -
PREPARED_DATA_FOLDER="../data/prepared/demand/"
mkdir -p ${PREPARED_DATA_FOLDER}
echo "Splitting datasets into train-valid-test..."
python3 scripts/datasets/demand/handle_demand.py ${INIT_DATA_FOLDER} ${PREPARED_DATA_FOLDER}
