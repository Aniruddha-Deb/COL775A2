set -e

python3 code/main.py train data/raw/train data/raw/validation data/processed/frames \
--save-name models/baseline \
--debug \
$@
