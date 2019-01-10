'''
Prints csv of series paths and their patient position
'''

import pydicom
import csv
import os

path = './data/cases/'
data_path = '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY'

with open('.data/cases/series.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        series_first = os.path.join(data_path, row[0], '000000.dcm')
        if os.path.isfile(series_first):
            ds = pydicom.dcmread(series_first)
            print(row[0] + ',' + row[1] + ',' + ds.PatientPosition)
