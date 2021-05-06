#!/bin/bash

# converts all-atom pdb files to different CG structures
cp protein_example.pdb protein_example_AA.pdb
python DeepCGSA.py -f protein_example.pdb -c CA
python DeepCGSA.py -f protein_example.pdb -c CACB
python DeepCGSA.py -f protein_example.pdb -c martini   #or use martinize.py directly
cp RNA_example.pdb RNA_example_AA.pdb
python DeepCGSA.py -f RNA_example.pdb -c P
python DeepCGSA.py -f RNA_example.pdb -c 3SPN

# predict SASA with DeepCGSA (for CG structures) or NACCESS (for all-atom structures)
python DeepCGSA.py -f protein_example_CA.pdb -t CA -o protein_CA
python DeepCGSA.py -f protein_example_CACB.pdb -t CACB -o protein_CACB
python DeepCGSA.py -f protein_example_martini.pdb -t martini -o protein_martini
python DeepCGSA.py -f RNA_example_P.pdb -t P -o RNA_P
python DeepCGSA.py -f RNA_example_3SPN.pdb -t 3SPN -o RNA_3SPN
python DeepCGSA.py -f protein_example_AA.pdb -t AA -o protein_AA
python DeepCGSA.py -f RNA_example_AA.pdb -t AA -o RNA_AA
