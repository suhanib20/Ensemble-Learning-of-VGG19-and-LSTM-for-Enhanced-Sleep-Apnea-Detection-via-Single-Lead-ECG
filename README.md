# Obstructive Sleep Apnea Detection from ECG using VGG19 + LSTM

This repository provides an end-to-end pipeline for detecting **Obstructive Sleep Apnea (OSA)** using **single-lead ECG signals** from the PhysioNet Apnea-ECG database. It includes signal preprocessing, feature extraction, and classification using a deep learning model combining **VGG19** and **LSTM** architectures.

---

## Project Structure:

- \`preprocessing.py\` : Data loading, R-peak detection, filtering, and feature extraction  
- \`model_training_and_evaluation.ipynb\` : VGG19 + LSTM model for classification and evaluation  
- \`apnea-ecg.pkl\` : Preprocessed dataset (can be generated)  
- \`dataset/osa_data/\` : Raw Apnea-ECG files (.dat, .hea, .apn)  

---

## Highlights:

- Uses biosppy and wfdb for accurate signal filtering and R-peak detection  
- Fully automated preprocessing pipeline with multiprocessing for scalability  
- Extracts R-R Interval (RRI) and Amplitude features from ECG  
- Combines VGG19 CNN with LSTM to capture both spatial and temporal ECG patterns  
- Robust training and evaluation split using official PhysioNet labels  

---

## Requirements:

Install dependencies using:

\`\`\`bash
pip install wfdb biosppy tqdm cpu_count
\`\`\`

Python version 3.6–3.10 recommended.

---

## Dataset:

Uses the Apnea-ECG database from PhysioNet, with 70 annotated recordings:  
- \`a01–a20\`, \`b01–b05\`, \`c01–c10\` for training  
- \`x01–x35\` for testing  

Each record contains a single-channel ECG signal and apnea event annotations (.apn files).  
Place dataset under \`dataset/osa_data/\`

---

## Preprocessing Pipeline:

Running \`preprocessing.py\` will:  
1. Load ECG signals and apnea labels  
2. Apply bandpass FIR filter (3–45 Hz)  
3. Detect and correct R-peaks using Hamilton segmenter + correction  
4. Compute R-R Intervals (RRI) and Amplitude at R-peaks  
5. Segment data into 5-minute windows (2 min before and after target minute)  
6. Save processed data as \`apnea-ecg.pkl\`  

---

## Model (VGG19 + LSTM):

- VGG19 CNN extracts hierarchical features from ECG feature maps  
- LSTM layers capture temporal dependencies across heartbeat sequences  
- Fully connected layers perform binary classification (Apnea or Normal)  

Run \`model_training_and_evaluation.ipynb\` to train, validate, and evaluate model using metrics: Accuracy, Precision, Recall, F1-score.

---

## Sample Results:

- Accuracy: ~91%  
- Precision: ~88%  
- Recall: ~89%  
- F1-Score: ~88.5%

---

## Citation
If you are using the above code, then please cite this manuscript:
Bhatia, S., Garg, D., Singh, H., Bansal, S. (2025). Ensemble Learning of VGG19 and LSTM for Enhanced Sleep Apnea Detection via Single-Lead ECG, The Visual Computer, Springer.



## Dataset Link:

https://physionet.org/content/apnea-ecg/1.0.0/

---


Thank you for your interest! Happy apnea detection!
