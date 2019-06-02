import numpy as np

from preprocess import Preprocessor
from metadata import Metadata
from tfrecords import TFRecordGenerator

reference_size = [(112 + 16) + 12, (88 + 16) + 12, (32 + 4) + 6]
k = 4
test_proportion = 0.25

data_path = '/vol/gpudata/rh2515/MRI_Crohns'
label_path = '/vol/gpudata/rh2515/MRI_Crohns/labels'
record_out_path = '/vol/gpudata/rh2515/MRI_Crohns/tfrecords/statistical_crop'
record_suffix = 'axial_t2_only_cropped'

abnormal_cases = list(range(50))
healthy_cases = list(range(50))
metadata = Metadata(data_path, label_path, abnormal_cases, healthy_cases, dataset_tag='')
# metadata = Metadata(data_path, label_path, abnormal_cases, healthy_cases, dataset_tag=' cropped')

for patient in reversed(metadata.patients):
    patient.load_image_data()

preprocessor = Preprocessor(constant_volume_size=reference_size)

metadata.patients = preprocessor.process(metadata.patients, ileum_crop=False, region_grow_crop=True, statistical_region_crop=True)

record_generator = TFRecordGenerator(record_out_path, record_suffix)
# record_generator.generate_train_test(test_proportion, metadata.patients)
record_generator.generate_cross_folds(k, metadata.patients)

print('Done')
