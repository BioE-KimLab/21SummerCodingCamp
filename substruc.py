import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from contextlib import closing
from matplotlib import pyplot as plt
import math

def substruc_search(smiles,radius=1):
    m = Chem.MolFromSmiles(smiles)
    substruc_smiles = []
    substruc_smarts = []
    
    for atom_index in np.arange(m.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(m,radius,int(atom_index))
        amap={}
        submol=Chem.PathToSubmol(m,env,atomMap=amap)
        #submol = Chem.AddHs(submol)
        mol_smarts=Chem.MolToSmarts(submol)
        mol_smiles=Chem.MolToSmiles(submol)
        
        substruc_smiles.append(mol_smiles)
        substruc_smarts.append(mol_smarts)
        
    return substruc_smiles, substruc_smarts


def db_fg_search(smi_all):
    sub_smi_all = []
    sub_sma_all = []
    '''
    #serial
    for smi in tqdm(smi_all):
        sub_smi, sub_sma = substruc_search(smi)
        sub_smi_all += sub_smi
        sub_sma_all += sub_sma
    '''

    #parallel
    import time
    start = time.time()
    from multiprocessing import Pool
    p = Pool(4)
    with closing(p):
        results = p.map(substruc_search, smi_all)
        p.terminate()
        p.join()

    sub_smi_all = [item for sublist in [x[0] for x in results] for item in sublist]
    sub_sma_all = [item for sublist in [x[1] for x in results] for item in sublist]

    end = time.time()
    print(end-start)

    #FG_count = Counter(sub_sma_all)
    #print(len(smi_all))
    #print(len(sub_sma_all))
    print(len(set(sub_sma_all)))
