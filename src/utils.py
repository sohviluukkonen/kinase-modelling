import os
import tqdm
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


def mkdirs(path : str):
    """Create a directory if it does not exist."""

    if not os.path.exists(path) : os.makedirs(path)

def compute_fps(data):
    """Compute Morgan Fingerprints from SMILES."""

    fps = pd.DataFrame(np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)

    return fps
