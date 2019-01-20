'''
Generates CSV of slices
'''

from data.handling.index_dataset import Indexer

indexer = Indexer('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
indexer.index_dataset('index.csv')
