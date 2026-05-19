from preprocessing import unpack_signal, form_subject_dict
import os

if __name__ == "__main__":
    """
    file_path = "./data/Training_WFDB/A6851"
    record = unpack_signal(file_path= file_path)
    print(record)
    """
    raw_path = "./data/raw"
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")
    di = form_subject_dict(signals_path=signals_path, ref_path=ref_path, subject_number=43 )
    print(di)