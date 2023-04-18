from src.inputs import create_input_files_from_pQSARdatasets
from src.pQSAR import pQSAR_model_validation, run_pQSAR_model

if __name__ == '__main__':

    # # Create input files from pQSAR datasets
    create_input_files_from_pQSARdatasets()

    # # Validate pQSAR implementation on Martin et al. dataset
    pQSAR_model_validation()

    # Run pQSAR model for both datasets, both splits with and without data leakage
    for dataset in ['kinase200','kinase1000']:
        for split in ['RGES', 'DGBC']:
            for data_leakage in ['Default', 'DataLeakage']:
                print(f'Running pQSAR model for {dataset} {split} {data_leakage}')
                run_pQSAR_model(dataset, split, data_leakage)
