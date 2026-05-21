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

def get_sig_lengths_list (signals_path, num_pat):
    lengths_list = []
    for subject_number in range(1, num_pat +1):
        #set file name
        subject_id = f"A{subject_number:04d}"
        #set signal file path
        sig_file_path = os.path.join(signals_path, subject_id)
        #get signals
        _, sig_len = unpack_signal(sig_file_path)
        lengths_list.append(sig_len)
    return lengths_list

def signal_lengths_analysis(lengths_list): 
    lengths = np.asarray(lengths_list)
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    mean= np.mean(lengths)

    #get bins
    bin1 = [length for length in lengths if 3000<=length<10000]
    bin2 = [length for length in lengths if 10000<=length<30000]
    bin3 = [length for length in lengths if 30000<=length<50000]
    bin4 = [length for length in lengths if 50000<=length<=72000]

if __name__ == "__main__":
    sig_path , _ , num_pat = set_data_paths()
    length_list = get_sig_lengths_list(sig_path, num_pat)
    signal_lengths_analysis(length_list)