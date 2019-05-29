FOLDS=$1

BASE="/vol/gpudata/rh2515"
RECORDS="MRI_Crohns/tfrecords/ti_n80_k4/axial_t2_only_cropped"
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
    echo "Running ${#@}-fold cross validation"

    for fold in ${@}
    do
      python3 run.py \
      Crohns_MRI \
      ${BASE}/${RECORDS}_train_fold${fold}.tfrecords \
      ${BASE}/${RECORDS}_test_fold${fold}.tfrecords \
      -record_shape 84,84,26 \
      -feature_shape 80,80,24 \
      -f=${fold} \
      -bS=48 \
      -lD=log_ti/${TIMESTAMP}/ \
      -nB=800
    done
fi
