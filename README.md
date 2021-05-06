# DeepCGSA---Accurate estimation of solvent accessible surface area for coarse-grained biomolecular structures with deep learning
We provide the python code of DeepCGSA implementation for estimating SASA based on CG structures.
## Requirements
```
numpy==1.19.4  
tensorflow==2.0.0  
pandas==1.0.3  
sklearn==0.23.1  
biopandas==0.2.7  
scipy==1.5.4  
biopython==1.78 (with NACCESS)
```
## Usage
```
Usage: DeepCGSA.py [options]
Options:
  -h, --help            show this help message and exit
  -f PDB_FILENAME, --file=PDB_FILENAME
                        The PDB file containing a CG protein or RNA structure
                        as discussed in paper. Please use same format as shown
                        in example input files (pdb format): example_CA.pdb,
                        example_CACB.pdb ......
  -t CG_TYPE, --CG_type=CG_TYPE
                        CG type of the PDB file, available for CA (Cα protein
                        structure), CACB (Cα-Cβ protein structure), martini
                        (Martini protein strcture), P (P-based RNA structure),
                        3SPN (3SPN RNA structure), AA (all-atom structure,
                        calculate by NACCESS). 
  -o OUTPUT_FILENAME, --output=OUTPUT_FILENAME
                        Residue-wise prediction will write to a csv file named
                        as output_filename.csv
  -c CREATE_CG, --create_CG_file=CREATE_CG
                        We provided a convenient function to create CG file
                        with appropriate format from AA file. -c option gave
                        the CG type to convert. When using -c option, script
                        will not calculate SASA but output a CG file. For
                        example: "python DeepCGSA.py -f example_AA.pdb -c CA".
                        Created file will be named as xxx_CGtype.pdb
```
`model_weight` should be copied to current work path with DeepCGSA.py
## Example
We provide input files with appropriate format: example_CA.pdb, example_CACB.pdb ......   
For example, run the following code to estimate SASA of the example Cα pdb file.
```
python DeepCGSA.py -f example_CA.pdb -t CA -o prediction
```
The "example.ipynb" provides a simple usage of DeepCGSA.





