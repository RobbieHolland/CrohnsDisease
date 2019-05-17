FOLDS=$1

BASE="/vol/gpudata/rh2515"
RECORDS="MRI_Crohns/tfrecords/cropped_k4/axial_t2_only_cropped"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`

if [ $# -eq 0 ]
  then
    echo "Single run"
    python3 run.py \
    Crohns_MRI \
    $BASE/${RECORDS}_train_.tfrecords \
    $BASE/${RECORDS}_test_.tfrecords \
    -record_shape 68,135,270 \
    -feature_shape 64,128,256 \
    -bS=48 \
    -lD=logdir_crohns/${TIMESTAMP}/ \
    -nB=250

  else
    echo "Running k-fold cross validation"

    for fold in $(seq 0 $(($FOLDS - 1)))
    do
      python3 run.py \
      Crohns_MRI \
      ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
      ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
      -record_shape 60,132,300 \
      -feature_shape 48,112,256 \
      -f=${fold} \
      -bS=24 \
      -lD=log_cropped/${TIMESTAMP}/ \
      -nB=300

    done
fi
