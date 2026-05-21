from preprocessing import segmentation
import numpy as np



def test_segmentation():
    signals = np.asarray([[1, 2],[3, 2],[5, 6], [3,5]])
    subj = {"subject_id" :  "12" , "signals" : signals , "labels" : "label"}
    di_list = segmentation(subj)
    print(di_list)

if __name__ == "__main__":
    test_segmentation()
