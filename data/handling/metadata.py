import random
import math
from functools import reduce

def split_records(records, train_test_split):
    random.shuffle(records)
    n_test = math.floor(train_test_split * sum([r.n_centroids() for r in records]))

    count, i = 0, 0
    test_records = []
    while (count < n_test):
        test_records.append(records[i])
        count += records[i].n_centroids()
        i += 1

    train_records = [r for r in records if r not in test_records]

    return train_records, test_records

class Metadata():
    def __init__(self, records=[]):
        self.records = records

    def add_records(self, records):
        self.records += records

    def n_records(self):
        return len(self.records)

    def abnormal_records(self):
        return [r for r in self.records if r.is_abnormal]

    def healthy_records(self):
        return [r for r in self.records if not r.is_abnormal]

    # Centroid count
    def n_abnormal_centroids(self):
        return sum([r.n_slice_centroids() for r in self.abnormal_records()])

    def n_healthy_centroids(self):
        return sum([r.n_slice_centroids() for r in self.healthy_records()])

    def centroid_histogram(self):
        return reduce(lambda a, b: a + b, [s.centroid_slice_proportion() for s in self.abnormal_records()])

    # Patient count
    def n_abnormal_patients(self):
        return len(set([r.patient_no for r in self.abnormal_records()]))

    def n_healthy_patients(self):
        return len(set([r.patient_no for r in self.healthy_records()]))

    # Statistics
    def print_statistics(self):
        print('Metadata statistics')
        print(len(self.records), 'records')
        print(self.n_abnormal_centroids(), 'centroids with polyps')
        print(self.n_healthy_centroids(), 'centroids without polyps')

    def filter_records(self, filter):
        self.records = [r for r in self.records if filter(r)]

    # One slice per healthy record (for now)
    def process_healthy_records(self, all_healthy_records):
        histogram = self.centroid_histogram()
        sampled_records = random.sample(all_healthy_records, self.n_abnormal_centroids())

        for i, record in enumerate(sampled_records):
            record.slice_centroids = [int(round(histogram[i] * record.volume_height))]
        self.add_records(sampled_records)

    def compute_neighbouring_slices(self, neighbour_distance):
        for record in self.records:
            all_slices = []

            for slice in record.slice_centroids:
                all_slices += list(set([nei for nei in range(slice - neighbour_distance, slice + neighbour_distance + 1)
                                        if nei >= 0 and nei < record.volume_height - 1]))
            record.slices = all_slices

    # Test train split
    def split(self, train_test_split):
        abnormal_train_records, abnormal_test_records = split_records(self.abnormal_records(), train_test_split)
        healthy_train_records, healthy_test_records = split_records(self.healthy_records(), train_test_split)

        train, test = abnormal_train_records + healthy_train_records, abnormal_test_records + healthy_test_records
        random.shuffle(train)
        random.shuffle(test)
        return Metadata(train), Metadata(test)

    def cut_dataset(self, n):
        self.records = self.records[1:(n + 1)]
