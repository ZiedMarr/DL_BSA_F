from preprocessing import segmentation, unpack_signal
import numpy as np



def test_segmentation():
    signals = np.asarray([[1, 2],[3, 2],[5, 6], [3,5]])
    subj = {"subject_id" :  "12" , "signals" : signals , "labels" : "label"}
    di_list = segmentation(subj)
    print(di_list)

def test_unpack_signal() : 
    file_path = "./data/raw/Training_WFDB/A6851"
    record, _  = unpack_signal(file_path= file_path)
    print(record.shape)

if __name__ == "__main__":
    test_unpack_signal()
