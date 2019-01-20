import random
import math
import csv
from data.handling.parse_labels import DataParser
from data.handling.slice_to_file_number import SliceToFile
from data.handling.study_record import header

class Indexer:
    def __init__(self, label_path, data_path):
        self.reader = DataParser(label_path, data_path)
        self.slice_to_file = SliceToFile(data_path)
        self.records = self.reader.read()
        self.abnormal = [r for r in self.records if r.is_abnormal]
        self.healthy =  [r for r in self.records if not r.is_abnormal]
        print(len(self.abnormal))
        print(len(self.healthy))

        random.seed(1234)

    def data_rows(self, record):
        rows = []
        for i, slice in enumerate(record.slices):
            rows.append(record.csv_rows())
        return rows

    # Loads image data
    def index_dataset(self, index_path):
        rows = []

        # Load abnormal slices
        print('Loading abnormal slice images...')
        for abnormal_record in self.abnormal:
            if len(abnormal_record.slices) == 0:
                continue
            print(abnormal_record.study_path)
            slice_files = self.slice_to_file.convert(abnormal_record)
            abnormal_record.set_slice_files(slice_files)
            rows += abnormal_record.csv_rows()


        # Load random set of healthy slices
        print('Loading healthy slice images...')
        random.shuffle(self.healthy)
        for i in range(len(rows)):
            healthy_record = self.healthy[i]
            print(healthy_record.study_path)
            slice = random.randint(0, healthy_record.volume_height - 1)
            healthy_record.slices = [slice]
            healthy_record.set_slice_files(self.slice_to_file.convert(healthy_record))
            rows += healthy_record.csv_rows()

        with open(index_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header())
            writer.writerows(rows)

        print('Dataset index ', index_path)
