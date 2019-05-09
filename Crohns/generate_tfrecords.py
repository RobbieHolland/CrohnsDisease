import numpy as np

from preprocess import Preprocessor
from metadata import Metadata
from tfrecords import TFRecordGenerator

reference_size = [256, 128, 64]
test_proportion = 0.2
data_path = '/vol/bitbucket/rh2515/MRI_Crohns'
record_out_path = '/vol/bitbucket/rh2515/MRI_Crohns/tfrecords'
record_suffix = 'axial_t2_only'

abnormal_cases = list(range(30))
healthy_cases = list(range(30))
metadata = Metadata(data_path, abnormal_cases, healthy_cases, dataset_tag='')
# metadata = Metadata(data_path, abnormal_cases, healthy_cases, dataset_tag=' cropped')

for patient in metadata.patients:
    patient.load_image_data()

preprocessor = Preprocessor(constant_volume_size=reference_size)

metadata.patients = preprocessor.process(metadata.patients)

record_generator = TFRecordGenerator(record_out_path, record_suffix)

labels = [patient.get_label() for patient in metadata.patients]
# record_generator.generate_train_test(test_proportion, features, labels)
record_generator.generate_cross_folds(5, metadata.patients)

print('Done')
