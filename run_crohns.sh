FOLDS=$1

for fold in $(seq 0 $(($FOLDS - 1)))
do
  python3 run.py \
  Crohns_MRI \
  /vol/bitbucket/rh2515/MRI_Crohns/tfrecords/axial_t2_only_train_fold$fold.tfrecords \
  /vol/bitbucket/rh2515/MRI_Crohns/tfrecords/axial_t2_only_test_fold$fold.tfrecords \
  64,128,256 \
  -bS=48 \
  -lD=logdir_crohns/fold_$fold \
  -nB=500
done
