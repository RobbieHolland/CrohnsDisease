FOLDS=$1

BASE="/vol/gpudata/rh2515"
RECORDS="MRI_Crohns/tfrecords/sc_localised/axial_t2_only_cropped"
# RECORDS="MRI_Crohns/tfrecords/sc_localised/axial_t2_only_cropped"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@}-fold cross validation"

for fold in ${@}
do
  python3 run.py \
  Crohns_MRI \
  ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
  ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
  -record_shape 42,112,120 \
  -feature_shape 32,80,96 \
  -f=${fold} \
  -bS=32 \
  -lD=log_sc_mixed/${TIMESTAMP}/ \
  -nB=500 \
  -at=1 \
  -ma=1 \
  -lc=0
done


# -record_shape 30,96,96 \
# -feature_shape 24,80,80 \
