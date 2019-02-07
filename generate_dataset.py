'''
Generates TFRecords files
'''

from data.handling.generate_tfrecord import TFRecordGenerator

generator = TFRecordGenerator('./data/cases/', '/vol/bitbucket/rh2515/CT_Colonography', 'data/tfrecords')
generator.generate_train_test(0.2)
