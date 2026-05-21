from preprocessing import unpack_signal, form_subject_dict
import os
import config 
import numpy as np

def set_data_paths() :
    #set paths
    conf = config.get_config()
    raw_path = conf["paths"]["raw_data"]
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")

    #set number of subjects
    num_pat = conf["raw_dataset"]["num_subjects"]

    return signals_path, ref_path, num_pat



def test_from_subject_dict():
    raw_path = "./data/raw"
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")
    di = form_subject_dict(signals_path=signals_path, ref_path=ref_path, subject_number=43 )
    print(di)

def test_unpack_signal() : 
    file_path = "./data/Training_WFDB/A6851"
    record = unpack_signal(file_path= file_path)
    print(record)




if __name__ == "__main__":
    pass




