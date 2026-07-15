from preprocessing import segmentation, unpack_signal, preprocessing_pipeline, form_subject_dict
import numpy as np
import torch
import os
from  matplotlib import pyplot as plt
from config import get_config
from transforms import stft_transform
from config import STFT_PARAMS

cfg = get_config()
sampling_rate = cfg["dataset"]["sampling_rate"]


def set_data_paths() :
    #set paths
    
    raw_path = cfg["paths"]["raw_data"]
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")

    #set number of subjects
    num_pat = cfg["raw_dataset"]["num_subjects"]

    return signals_path, ref_path, num_pat

signals_path, ref_path, num_pat = set_data_paths()

def test_segmentation():
    signals = np.asarray([[1, 2],[3, 2],[5, 6], [3,5]])
    subj = {"subject_id" :  "12" , "signals" : signals , "labels" : "label"}
    di_list = segmentation(subj)
    print(di_list)

def test_unpack_signal() : 
    file_path = "./data/raw/Training_WFDB/A6851"
    record, _  = unpack_signal(file_path= file_path)
    print(record.shape)

def visualize_signal(signal, channel, preprocess=False):
    title = "raw"
    if preprocess:
        signal = preprocessing_pipeline(signal=signal)
        title = "preprocessed"
    t = np.arange(len(signal[:,channel])) / sampling_rate
    plt.figure()
    plt.plot(t, signal[:, channel])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.show(block= False)

if __name__ == "__main__":
    """
    subj = form_subject_dict(signals_path, ref_path, 1222)
    visualize_signal(subj["signals"], 11, True)
    visualize_signal(subj["signals"], 11, False)
    plt.pause(0.001)
    input("Press Enter to close...")
    """
    subj = form_subject_dict(signals_path, ref_path, 1222)
    subj["signals"] = preprocessing_pipeline(signal=subj["signals"])
    segments = segmentation(subject_dict=subj)

    sample_signal = torch.tensor(segments[0]["signals"], dtype=torch.float32)  # (12, segment_length)
    transform = stft_transform(**STFT_PARAMS)
    out = transform(sample_signal)
    print(out.shape)  # (12, F, T)

