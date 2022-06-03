from substruc import db_fg_search
import pandas as pd

bde_db = pd.read_csv('rdf_data_190531.csv')
db_fg_search(list(bde_db.molecule.unique()))

