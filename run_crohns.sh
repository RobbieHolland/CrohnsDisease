FOLDS=$1

BASE="/vol/bitbucket/rh2515"
RECORDS="MRI_Crohns/tfrecords/ti_imb/axial_t2_only"
# RECORDS="MRI_Crohns/tfrecords/statistical_crop/axial_t2_only_cropped"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@}-fold cross validation"

for fold in ${@}
do
  python3 run.py \
  Crohns_MRI \
  ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
  ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
  -record_shape 37,99,99 \
  -feature_shape 31,87,87 \
  -f=${fold} \
  -bS=64 \
  -lD=log_ti/${TIMESTAMP}/ \
  -nB=900
done

# -record_shape 42,116,140 \
# -feature_shape 32,88,112 \
