import csv
import os
import pydicom
from data.handling.study_record import StudyRecord

# Slices are of the form 'Pos >=10mm 34/45/63/109'
def parse_slices(slices):
    if '/' in slices:
        return [int(slice) for slice in
                slices.replace('Pos >=10mm', '').split('/')]
    return []

def parse_abnormal_row(row):
    # [Case ID, [supine slice numbers], [prone slice numbers]]
    return [row[0], parse_slices(row[1]), parse_slices(row[2])]

def parse_path_position(row):
    # Patient no, Path, Volume height, Patient position
    return (row[0].split('/')[1], row[0], row[1], row[2])

def parse_slice_to_file(row):
    # Study path, slice number, file number
    return (row[0], int(row[1]), int(row[2]))

def read_csv(path, file, row_parser, header=True):
    parsed_data = []
    with open(os.path.join(path, file + '.csv'), 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data = row_parser(row)
            parsed_data.append(data)
        # Discard column headers
        if header:
            return parsed_data[1:]
        return parsed_data

class DataParser:
    def __init__(self, label_path, data_path):
        self.label_path = label_path
        self.data_path = data_path
        self.relevant_positions = [['FFS', 'HFS'], ['FFP', 'HFP']]

    def read(self):
        records = []
        studies = read_csv(self.label_path, 'studies_positions', parse_path_position, header=False)
        polyps =   [read_csv(self.label_path, 'None', lambda x: [x[0], [], []]),
                    read_csv(self.label_path, '6-9', parse_abnormal_row),
                    read_csv(self.label_path, '10+', parse_abnormal_row)]

        for class_number, polyp_class in enumerate(polyps):
            for record in polyp_class:
                patient_no = record[0]
                matching_studies = [s for s in studies if s[0] == patient_no]

                relevant_records = []
                for i, position in enumerate(self.relevant_positions):
                    polyp_slices = record[i + 1]

                    next_records = [StudyRecord(patient_no, s[1], polyp_slices, int(s[2]), s[3], class_number)
                                       for s in matching_studies if s[3] in position]
                    if len(next_records) == 1:
                        relevant_records += next_records
                records += relevant_records

        return records
