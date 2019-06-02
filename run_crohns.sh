FOLDS=$1

BASE="/vol/gpudata/rh2515"
RECORDS="MRI_Crohns/tfrecords/statistical_crop/axial_t2_only_cropped"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@}-fold cross validation"

for fold in ${@}
do
  python3 run.py \
  Crohns_MRI \
  ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
  ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
  -record_shape 42,116,140 \
  -feature_shape 32,88,112 \
  -f=${fold} \
  -bS=74 \
  -lD=log_statistical_crop/${TIMESTAMP}/ \
  -nB=800
done
