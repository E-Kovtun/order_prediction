export PYTHONPATH=.
INIT_DATA_FOLDER="../data/initial/order/"
mkdir -p ${INIT_DATA_FOLDER}
PREPARED_DATA_FOLDER="../data/prepared/order/"
mkdir -p ${PREPARED_DATA_FOLDER}
echo "Splitting datasets into train-valid-test..."
python3 scripts/datasets/order/handle_order.py ${INIT_DATA_FOLDER} ${PREPARED_DATA_FOLDER}
