# Sesotho Code-Switched Transcription Analysis
Project Overview
This project evaluates the performance of two automatic speech recognition (ASR) models, Whisper-Small and Wav2Vec2, on a code-switched Sesotho dataset. It includes:

Exploratory Data Analysis (EDA)
Model Evaluation (EMA) - Benchmarking WER, CER
Error Analysis (EA) - Visualizing errors & linguistic insights
Recommendations for Improvement

# 1️⃣ Environment Setup
You can run this project on Google Colab or locally. If running locally, install dependencies:

# A. Using Google Colab
1. Upload the notebook (sesotho_transcription_analysis.ipynb) to Google Colab.
2. Mount Google Drive if required:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
3. Install dependencies (required libraries):
```bash
!pip install transformers datasets jiwer torchaudio librosa phonemizer seaborn
```
# B. Running Locally
1. Clone this repository
```bash
git clone https://github.com/your-repo/sesotho-asr-evaluation.git
cd sesotho-asr-evaluation
```
2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. Install required packages:
```bash
pip install -r requirements.txt
```
# 2️⃣ Running the Notebook
1. Open Jupyter Notebook:
```bash
jupyter notebook
```
2. Load sesotho_transcription_analysis.ipynb.
3. Run all cells sequentially.
# 3️⃣ Dataset
The dataset consists of 97 Sesotho code-switched audio samples (.wav) with transcriptions in transcriptions.csv.
# Loading Data in Notebook
```bash
import pandas as pd

# Load transcriptions
df = pd.read_csv("path/to/transcriptions.csv")
df.head()
```
# 4️⃣ Reproducing Results
A. Model Evaluation (EMA)
To compare Whisper-Small and Wav2Vec2, we measure:
  * Character Error Rate (CER)
  * Word Error Rate (WER)
# Running Model Benchmarking
```bash
from jiwer import wer, cer

wer_score = wer(ground_truth, predictions)
cer_score = cer(ground_truth, predictions)

print(f"WER: {wer_score:.2f}, CER: {cer_score:.2f}")
```
# B. Error Analysis (EA)
1. Confusion Matrix of Errors
```bash
import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix of common transcription mistakes
sns.heatmap(error_matrix, annot=True, cmap="Blues")
plt.title("Word-Level Confusion Matrix")
plt.show()
```
2. Phoneme-Level Error Analysis
```bash
import phonemizer
phonemes_gt = phonemizer.phonemize(ground_truth, language="en")
phonemes_pred = phonemizer.phonemize(predictions, language="en")

print(f"Phoneme-Level WER: {wer(phonemes_gt, phonemes_pred):.2f}")
```
# 5️⃣ Key Findings & Recommendations
✅ Whisper-Small performed better than Wav2Vec2 on Sesotho code-switched speech.
✅ Common errors: phoneme misinterpretations, missing words, and long-audio degradation.
✅ Recommended Improvements:

* Train on larger, domain-specific Sesotho datasets.
* Use phoneme-aware transcription models.
* Apply post-processing correction using a language model.

# 6️⃣ Citation & Credits
  * Dataset Source: [NGO in Lesotho]
  * Pretrained Models: [OpenAI Whisper, Facebook Wav2Vec2]

