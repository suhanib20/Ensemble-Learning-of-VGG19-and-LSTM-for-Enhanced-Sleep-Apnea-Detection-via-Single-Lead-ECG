# Install required packages
!pip install wfdb biosppy
 
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
 
# Imports
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from multiprocessing import cpu_count
from tqdm import tqdm
 
# Constants
base_dir = "/content/drive/MyDrive/dataset/osa_data"
fs = 100  # Sampling frequency
sample = fs * 60  # Samples per minute
before = 2  # Minutes before
after = 2   # Minutes after
hr_min = 20
hr_max = 300
num_worker = min(cpu_count() - 1, 35) if cpu_count() > 1 else 1
 
# Worker function
def worker(name, labels):
    X = []
    y = []
    groups = []
    try:
        signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0]
    except Exception as e:
        print(f"Error reading signal for {name}: {e}")
        return X, y, groups
 
    for j in tqdm(range(len(labels)), desc=name, file=sys.stdout):
        if j < before or (j + 1 + after) > len(signals) / float(sample):
            continue
        signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
        try:
            signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
                                            frequency=[3, 45], sampling_rate=fs)
            rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
            rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
            if len(rpeaks) / (1 + after + before) < 40 or len(rpeaks) / (1 + after + before) > 200:
                continue
            rri_tm, rri_signal = rpeaks[1:] / float(fs), np.diff(rpeaks) / float(fs)
            rri_signal = medfilt(rri_signal, kernel_size=3)
            ampl_tm, ampl_signal = rpeaks / float(fs), signal[rpeaks]
            hr = 60 / rri_signal
            if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
                X.append([(rri_tm, rri_signal), (ampl_tm, ampl_signal)])
                y.append(0. if labels[j] == 'N' else 1.)
                groups.append(name)
        except Exception as e:
            print(f"Error processing {name} segment {j}: {e}")
            continue
    return X, y, groups
 
# Main Execution
if __name__ == "__main__":
    apnea_ecg = {}
 
    train_names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]
 
    o_train = []
    y_train = []
    groups_train = []
    print('Training...')
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for name in train_names:
            try:
                labels = wfdb.rdann(os.path.join(base_dir, name), extension="apn").symbol
                task_list.append(executor.submit(worker, name, labels))
            except Exception as e:
                print(f"Error reading annotation for {name}: {e}")
        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_train.extend(X)
            y_train.extend(y)
            groups_train.extend(groups)
 
    # Read test answers
    answers = {}
    with open("/content/drive/MyDrive/dataset/event-2-answers.txt", "r") as f:
        for answer in f.read().strip().split("\n\n"):
            key = answer[:3]
            symbols = list("".join(answer.split()[2::2]))
            answers[key] = symbols
 
    test_names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]
 
    o_test = []
    y_test = []
    groups_test = []
    print("Testing...")
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for name in test_names:
            try:
                labels = answers[name]
                task_list.append(executor.submit(worker, name, labels))
            except Exception as e:
                print(f"Error getting labels for {name}: {e}")
        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_test.extend(X)
            y_test.extend(y)
            groups_test.extend(groups)
 
    # Save the dataset
    apnea_ecg = dict(
        o_train=o_train,
        y_train=y_train,
        groups_train=groups_train,
        o_test=o_test,
        y_test=y_test,
        groups_test=groups_test
    )
 
    with open(os.path.join(base_dir, "apnea-ecg.pkl"), "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)
 
    print("done")  # Dataset used is Apnea-ECG Database from PhysioNet (publicly available)
