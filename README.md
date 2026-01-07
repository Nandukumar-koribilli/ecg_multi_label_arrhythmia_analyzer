# CardiAI - ECG Multi-Label Arrhythmia Analyzer (Streamlit Version)

A Streamlit web application for analyzing ECG signals using Google's Gemini AI for multi-label arrhythmia classification.

## Features

- 📁 Batch upload of CSV ECG files
- 📊 Interactive ECG waveform visualization with Plotly
- 🤖 AI-powered arrhythmia detection using Gemini 1.5 Pro
- 📈 Comprehensive analysis dashboard with vitals and clinical insights
- 🔄 Batch processing capabilities

## Setup

1. Clone or download this repository
2. Navigate to the `streamlit_app` directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your Gemini API key:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Update the `.env` file with your API key:
     ```
     API_KEY=your_actual_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```

   Or if streamlit is in your PATH:
   ```bash
   streamlit run app.py
   ```

2. Upload CSV files containing ECG data (12-lead format)
3. Select leads to visualize the ECG waveforms
4. Click "Analyze This File" for individual analysis or "Analyze All Files" for batch processing
5. View the AI-generated analysis results including:
   - Detected arrhythmias with confidence scores
   - Vital signs (heart rate, QRS duration, QT interval)
   - Clinical summary and insights

## CSV Format

The application expects CSV files with the following format:
- Columns: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 (ECG leads)
- Optional: time column (will be auto-generated if missing)
- Values: ECG voltage measurements

## Supported Arrhythmias

- AF (Atrial Fibrillation)
- LBBB (Left Bundle Branch Block)
- Normal Heartbeat
- PAC (Premature Atrial Contraction)
- PVC (Premature Ventricular Contraction)
- RBBB (Right Bundle Branch Block)
- STD (ST-segment Depression)
- STE (ST-segment Elevation)

## Disclaimer

**STRICTLY FOR CLINICAL RESEARCH USE ONLY**

This tool is designed for research purposes and should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals for patient care.