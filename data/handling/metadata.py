import random
import math

class Metadata():
    def __init__(self):
        self.records = []

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

    # Slice count
    def n_abnormal_slices(self):
        return sum([r.n_slices() for r in self.abnormal_records()])

    def n_slices(self):
        return sum([r.n_slices() for r in self.records])

    def n_healthy_slices(self):
        return self.n_slices() - self.n_abnormal_slices()

    # Patient count
    def n_abnormal_patients(self):
        return len(set([r.patient_no for r in self.abnormal_records()]))

    def n_healthy_patients(self):
        return len(set([r.patient_no for r in self.healthy_records()]))

    # Statistics
    def print_statistics(self):
        print(len(self.records), 'records')
        print(self.n_abnormal_slices(), 'slices with polyps')
        print(self.n_healthy_slices(), 'slices without polyps')

    def filter_records(self, filter):
        self.records = [r for r in self.records if filter(r)]

    # ==== Add neighbouring slices
    # For healthy slices, first allocates random locations
    # Then for all slices include neighbouring slices
    def compute_slices(self, slices_per_healthy, neighbour_distance):
        for record in self.records:
            all_slices = []
            if not record.is_abnormal:
                record.slice_centroids = random.sample(range(100, record.volume_height - 100), slices_per_healthy)

            for slice in record.slice_centroids:
                all_slices += list(set([nei for nei in range(slice - neighbour_distance, slice + neighbour_distance + 1)
                                        if nei >= 0 and nei < record.volume_height - 1]))
            record.slices = all_slices

    # Test train split
    def split(self, train_test_split):
        n_test = math.floor(train_test_split * self.n_records())
        test_indicies = random.sample(range(self.n_records()), n_test)
        test_records = [self.records[i] for i in test_indicies]
        train_records = [r for r in self.records if r not in test_records]

        return train_records, test_records

    def cut_dataset(self, n):
        self.records = self.records[1:(n + 1)]
