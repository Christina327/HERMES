import os
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem

from utils import get_MACCS
from drug_util import drug_feature_extract

def getData(dataset, use_disease=False):
    if dataset == 'ONEIL':
        dataset_name = 'ONEIL-COSMIC'
    elif dataset == "ALMANAC":
        dataset_name = 'ALMANAC-COSMIC'
    else:
        raise NotImplementedError
    DATASET_DIR = '../data'
    DISEASE_DIR = "DRUG-DISEASE"
    drug_smiles_file = os.path.join(DATASET_DIR, dataset_name, 'drug_smiles.csv')
    cline_feature_file = os.path.join(DATASET_DIR, dataset_name, 'cell line_gene_expression.csv')
    drug_synergy_file = os.path.join(DATASET_DIR, dataset_name, 'drug_synergy.csv')
    disease_cuis_file = os.path.join(DATASET_DIR, DISEASE_DIR, 'disease_cuis.csv')
    disease_feat_file = os.path.join(DATASET_DIR, DISEASE_DIR, 'disease_embd.npy')
    drug_interaction_file = os.path.join(DATASET_DIR, DISEASE_DIR, 'drug_disease_indication.csv')

    featurizer = dc.feat.ConvMolFeaturizer()
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data, drug_smiles_fea = pd.DataFrame(), list()
    for pubchemid, isosmiles in zip(drug['pubchemid'], drug['isosmiles']):

        mol = Chem.MolFromSmiles(isosmiles)
        mol_f = featurizer.featurize(mol)
        drug_data[str(pubchemid)] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
        drug_smiles_fea.append(get_MACCS(isosmiles))

    drug_fea = drug_feature_extract(drug_data)
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_fea = np.array(gene_data, dtype='float32')


    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)

    drug_num = len(drug_data.keys())
    drug_map = dict(zip(drug_data.keys(), range(drug_num)))
    cline_num = len(gene_data.index)
    cline_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    synergy = [
        [
            drug_map[str(row[0])],
            drug_map[str(row[1])],
            cline_map[row[2]],
            float(row[3])
        ]
        for _, row in synergy_load.iterrows()
        if (str(row[0]) in drug_data.keys()
            and str(row[1]) in drug_data.keys()
            and str(row[2]) in gene_data.index)
    ]

    if use_disease:
        disease_fea = np.load(disease_feat_file)
        disease_cuis = pd.read_csv(disease_cuis_file, index_col=0)
        disease_num = len(disease_cuis)
        disease_map = dict(zip(
            disease_cuis.index, 
            range(drug_num+cline_num, drug_num+cline_num+disease_num)))
        interaction_load = pd.read_csv(drug_interaction_file)
        interaction = [
            [
                drug_map[str(pubchemid)],
                disease_map[cuis]
            ]
            for _, (pubchemid, cuis, _) in interaction_load.iterrows()
            if str(pubchemid) in drug_data.keys()
        ]
    
    ret = (drug_fea, drug_smiles_fea, cline_fea, gene_data, synergy)
    if use_disease:
        ret += (disease_fea, interaction)
    
    return ret