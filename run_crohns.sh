python3 run.py \
Crohns_MRI \
/vol/bitbucket/rh2515/CrohnsDisease/Crohns/tfrecords/axial_t2_only_train.tfrecords \
/vol/bitbucket/rh2515/CrohnsDisease/Crohns/tfrecords/axial_t2_only_test.tfrecords \
64,128,256 \
-bS=48 \
-lD=logdir_crohns
