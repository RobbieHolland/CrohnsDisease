FOLDS=$1

BASE="/vol/gpudata/rh2515"
RECORDS="MRI_Crohns/tfrecords/ti_n100_k4_large/axial_t2_only_cropped"
# RECORDS="MRI_Crohns/tfrecords/sc_localised/axial_t2_only_cropped"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@}-fold cross validation"

for fold in ${@}
do
  python3 run.py \
  Crohns_MRI \
  ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
  ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
  -record_shape 30,96,96 \
  -feature_shape 24,80,80 \
  -f=${fold} \
  -bS=74 \
  -lD=log_ti_n100_large_TEST/${TIMESTAMP}/ \
  -nB=1000 \
  -at=0 \
  -lc=0
done


# -record_shape 30,96,96 \
# -feature_shape 24,80,80 \
