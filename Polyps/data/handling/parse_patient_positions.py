'''
Prints csv of study paths and their patient position
'''

import pydicom
import csv
import os

path = './data/cases/'
data_path = '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY'

with open('./data/cases/studies.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        studies_first = os.path.join(data_path, row[0], '000000.dcm')
        if os.path.isfile(studies_first):
            ds = pydicom.dcmread(studies_first, stop_before_pixels=True)
            print(row[0] + ',' + row[1] + ',' + ds.PatientPosition)
