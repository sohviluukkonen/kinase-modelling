from src.inputs import *

if __name__ == '__main__':

    # Create input files from kinase datasets
    for dataset in ['kinase200', 'kinase1000']:
        for split in ['RGES', 'DGBC']:
            create_input_files(f'data/datasets/{dataset}_{split}.csv.gz', 
                f'ModelInputs/{dataset}/{split}/Original')

    # Create input files from pQSAR datasets
    create_input_files_from_pQSARdatasets()

