# preprocessing.py

import os
import numpy as np
import wfdb
import pandas as pd
from config import get_config

cfg = get_config()


def get_labels(csv_path, idx):
    """
    for an index (corresponds to a subject), returns the labels as a list
    """
    df = pd.read_csv(csv_path)
    #get the labels
    row = df.iloc[idx-1]
    labels = row[["First_label", "Second_label", "Third_label"]].dropna().astype(int).tolist()
    
    return  labels


def unpack_signal(file_path):
    """
    parameters: 
        file_path : path of the .hea file without .hea
    returns:
        signals from the .mat file as an np array of shape (C, N)
    """
        
    # Load record 
    record = wfdb.rdrecord(file_path)
    # Access the signal data as a NumPy array
    signals = record.p_signal        # physical units (mV, etc.)
    # signals = record.d_signal      # digital (raw) values
    # Metadata from the .hea file
    sig_len = record.sig_len  # number of samples

    return signals, sig_len

def form_subject_dict(signals_path, ref_path, subject_number):
    #TODO : change signal shape (C,N) -> (N,C)
    """
    returns: 
        a directory that contains : subject_id, signals, labels
    """
    #set file name
    subject_id = f"A{subject_number:04d}"
    #set signal file path
    sig_file_path = os.path.join(signals_path, subject_id)
    #get signals
    signals, _ = unpack_signal(sig_file_path)
    #get labels
    labels = get_labels(ref_path, subject_number)
    # form subject dict:
    subject_dict = { "subject_id" :  subject_id , "signals" : signals , "labels" : labels}
        
    return subject_dict

def preprocess_data(config):
    #set paths
    raw_path = config["paths"]["raw_data"]
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")

    #set number of subjects
    num_pat = config["raw_dataset"]["num_subjects"]

    for i in range(1, num_pat +1):
        subject_dict = form_subject_dict(signals_path=signals_path, ref_path=ref_path, subject_number=i)
        # preprocess the signals
        subject_dict["signals"] = preprocessing_pipeline(subject_dict["signals"])
        #save file
        save_subject_file(subject_dict)


def save_subject_file(subject_dict):
    #TODO: implement a method that saves the subject_dict as a file ( choose the right format)
    pass
def preprocessing_pipeline(signals):
    #TODO: implement preprocessing pipeline
    pass


def preprocess_dataset(config):
    """
    Preprocessing pipeline.

    Students should:
    - load raw data (BIDS or other format)
    - apply filtering / normalization
    - segment signals into (N, C, T)

    Each saved file MUST contain:
        {
            "signals": (N, C, T),
            "labels": (N,),
            "subject_id": str or int
        }

    One file can represent:
    - a run
    - a session
    - or any logical chunk
    """

    raw_path = config["paths"]["raw_data"]
    save_path = config["paths"]["processed_data"]

    os.makedirs(save_path, exist_ok=True)

    print("Running preprocessing...")

    file_counter = 0

    # -------------------------
    # Example structure (BIDS-like)
    # -------------------------
    for subject in os.listdir(raw_path):

        if not subject.startswith("sub-"):
            continue

        subject_id = subject
        subject_path = os.path.join(raw_path, subject)

        for root, _, files in os.walk(subject_path):
            for file in files:

                # -------------------------
                # TODO: filter valid data files (e.g. .snirf, .edf)
                # -------------------------

                file_path = os.path.join(root, file)

                # -------------------------
                # TODO: load real data here
                # -------------------------
                signals = np.random.randn(50, 1, 1000)
                labels = np.random.randint(0, 2, 50)

                save_name = f"sample_{file_counter}.npy"

                np.save(os.path.join(save_path, save_name), {
                    "signals": signals,
                    "labels": labels,
                    "subject_id": subject_id
                })

                file_counter += 1

    print("Preprocessing complete.")


def segmentation(subject_dict : dict):
    """
    segments the signals of a subject. Segment_size : to be found in config

    parameteres:
        subject_dict: gets a subject directory, with "signals" : (N, C)
    returns: 
        dicts_list: a list containing dictionnaries with "signals": segment of the original signal (N, C)
    
    """
    dicts_list = []
    
    signals = subject_dict["signals"]
    #define number of number of segments
    segment_size = cfg["dataset"]["segment_length"]
    n_segments = signals.shape[0] // segment_size
    #split signal array into equally sized 
    segments_list = [signals[ i*segment_size:(i+1)*segment_size, : ] for i in range(n_segments) ] 

    for segment in segments_list:
        data_point = {"subject_id" :  subject_dict["subject_id"] , "signals" : segment , "labels" : subject_dict["labels"]}
        dicts_list.append(data_point)
    
    return dicts_list






