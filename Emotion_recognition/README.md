# 🎤 Emotion Recognition from Speech (ERS)

This project aims to classify human emotions (like **happy**, **angry**, **sad**, etc.) from audio recordings using deep learning techniques. It uses MFCC-based feature extraction along with a hybrid **CNN + LSTM** model to capture both spatial and temporal aspects of speech signals.

---

## 🚀 Project Pipeline

1. **Dataset Preparation**
   - Datasets used: RAVDESS, TESS, and EMO-DB
   - Labels: `Happy`, `Angry`, `Sad`, `Neutral`, etc.
   - Oversampling for balanced class distribution

2. **Feature Extraction**
   - Extracted **MFCCs** using `librosa`
   - Padded/truncated features for uniform shape
   - Scaled using `StandardScaler`

3. **Data Splitting**
   - 80% Training, 10% Validation, 10% Testing
   - Stratified sampling for label distribution

4. **Model Architecture**
   - 1D CNN Layers for spatial feature extraction
   - LSTM Layers for capturing temporal dependencies
   - BatchNorm, Dropout for regularization
   - Final Dense Layer with `softmax` activation

5. **Training Strategy**
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Callbacks:
     - EarlyStopping
     - ReduceLROnPlateau
     - ModelCheckpoint

6. **Evaluation**
   - Classification Report
   - Confusion Matrix
   - Accuracy on Test Data

7. **Real-Time Prediction**
   - Load a trained `.keras` model
   - Predict emotion from any given audio file path

---

## 📈 Results

- ✅ **Best Validation Accuracy**: ~81%
- 📊 **Test Accuracy**: ~79–82% depending on dataset
- 🔥 Model Generalizes well to unseen data

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

Key Libraries:
librosa

numpy, pandas

tensorflow, keras

scikit-learn

matplotlib, seaborn
