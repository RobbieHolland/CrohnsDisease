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

        self.n_polyps = sum([len(a.slices) for a in self.abnormal])

        random.seed(1234)

    def data_rows(self, record):
        rows = []
        for i, slice in enumerate(record.slices):
            rows.append(record.csv_rows())
        return rows

    def find_neighbouring_slices(self, slices):
        neighbouring_slices = []
        for slice in slices:
            neighbouring_slices += [slice - 1, slice, slice + 1]
        return neighbouring_slices

    def set_neighbouring_slice_files(self, record):
        neighbouring_slices = self.find_neighbouring_slices(record.slices)
        slices, slice_files = self.slice_to_file.convert(record, neighbouring_slices)
        record.slices = slices
        record.slice_files = slice_files
        return record

    # Loads image data
    def index_dataset(self, index_path):
        rows = []

        # Load abnormal slices
        print('Loading abnormal slice images...')
        for abnormal_record in self.abnormal:
            if len(abnormal_record.slices) == 0:
                continue
            print(abnormal_record.study_path)
            record = self.set_neighbouring_slice_files(abnormal_record)
            rows += abnormal_record.csv_rows()

        # Load random set of healthy slices
        print('Loading healthy slice images...')
        random.shuffle(self.healthy)
        for i in range(self.n_polyps):
            healthy_record = self.healthy[i]
            print(healthy_record.study_path)

            slice = random.randint(0, healthy_record.volume_height - 1)
            healthy_record.slices = [slice]

            record = self.set_neighbouring_slice_files(healthy_record)
            rows += healthy_record.csv_rows()

        with open(index_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header())
            writer.writerows(rows)

        print('Dataset index ', index_path)
