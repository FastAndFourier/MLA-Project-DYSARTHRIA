import librosa
import numpy as np
import os

PATH = "data/"

def set_fromdir(dir_target):
    labels = []
    records = []
    
    for dir in dir_target:
        sessions = [subdir for subdir in os.listdir(PATH+dir) if subdir.startswith('Session')]
        for session in sessions:
            if "wav_arrayMic" in os.listdir(PATH+dir+'/'+session):
                for file in os.listdir(PATH+dir+'/'+session+"/wav_arrayMic"):
                    y, _ = librosa.load(PATH+dir+'/'+session+"/wav_arrayMic/"+file, sr = 16000, duration=2.5)
                    
                    if len(y) == 0:
                        continue
                    
                    if len(y) < 40000 :
                        "Padding with signal repetition"
                        y = librosa.util.fix_length(y, 40000, mode = "wrap")
                    
                    label = 0 if "C" in dir else 1
                    labels.append(label)
                    records.append(y)
    
    labels = np.array(labels)
    records = np.array(records)
    return records, labels

def create_trainset():
    dir_target = ["FC02", "F03", "F01", "MC04", "MC03", "M02"]
    
    records, labels = set_fromdir(dir_target)
    
    np.save("data/x_train", records)
    np.save("data/y_train", labels)
    
    print(f"Save train set done,\n {labels.shape = }\n {records.shape = }")

def create_valset():
    dir_target = ["MC02", "FC01", "M03", "M01"]
    
    records, labels = set_fromdir(dir_target)
    
    np.save("data/x_val", records)
    np.save("data/y_val", labels)
    
    print(f"Save val set done,\n {labels.shape = }\n {records.shape = }")
    

def create_testset():
    dir_target = ["FC03", "F04", "MC01", "M05", "M04"]   
    
    records, labels = set_fromdir(dir_target)
    
    np.save("data/x_test", records)
    np.save("data/y_test", labels)
    
    print(f"Save test set done,\n {labels.shape = }\n {records.shape = }")
          
                    
if __name__ == "__main__":
    create_trainset()
    create_valset()
    create_testset()