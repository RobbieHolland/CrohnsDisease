# Prints the list of cases in some given categories
# Categories: None, 6-9, 10+

import os
import fileinput

path = '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY/'
cases = [str(l.rstrip()) for l in fileinput.input()]

for case in cases:
    case_path = os.path.join(path, case)
    print('------' + case)
    skip = os.listdir(case_path)[0]
    subdir_path = os.path.join(case_path, skip)
    subdirs = os.listdir(subdir_path)
    print(subdirs)
