import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# Constants
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def parse_csv(file) -> pd.DataFrame:
    """Parse uploaded CSV file into DataFrame."""
    df = pd.read_csv(file)
    # Add time column if not present
    if 'time' not in df.columns:
        df['time'] = [i * 2 for i in range(len(df))]  # Assuming 2ms intervals
    return df.head(1000)  # Limit for display

def plot_ecg(data: pd.DataFrame, active_leads: List[str]):
    """Create ECG plot using Plotly."""
    fig = go.Figure()

    colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b',
              '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
              '#f43f5e', '#6366f1', '#14b8a6', '#f97316']

    for i, lead in enumerate(active_leads):
        if lead in data.columns:
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=data[lead],
                mode='lines',
                name=f'Lead {lead}',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))

    fig.update_layout(
        title="ECG Waveform",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (mV)",
        height=400,
        showlegend=True
    )

    return fig

def analyze_ecg_signal(signal_sample: str, file_name: str) -> Dict[str, Any]:
    """Analyze ECG signal using Gemini AI."""
    prompt = f"""
    Analyze the following 12-lead ECG signal statistical summary and segments extracted from file "{file_name}".

    Task: Multi-label classification for the following classes:
    AF (Atrial Fibrillation), LBBB (Left Bundle Branch Block), Normal Heartbeat,
    PAC (Premature Atrial Contraction), PVC (Premature Ventricular Contraction),
    RBBB (Right Bundle Branch Block), STD (ST-segment Depression), STE (ST-segment Elevation).

    Rules for Output:
    1. If the patient has multiple simultaneous arrhythmias, list all of them.
    2. If the patient is 'Normal Heartbeat', do NOT include any other arrhythmia labels.
    3. Ignore the 'Others' classification in the final visible list unless no other patterns are detected.
    4. Provide two types of Clinical Interpretations:
       - If Normal: Provide high-level health maintenance advice (e.g., diet, exercise, stress management).
       - If Diseased (any arrhythmia detected): Provide specific clinical insights, pathophysiology, and potential urgency.

    Data Context:
    ECG Metadata/Segment Sample:
    {signal_sample}

    Return the response in JSON format with the following structure:
    {{
        "predictions": [
            {{
                "label": "string",
                "confidence": number,
                "description": "string"
            }}
        ],
        "summary": "string",
        "clinicalInsights": ["string"],
        "vitals": {{
            "bpm": number,
            "qrsDuration": "string",
            "qtInterval": "string"
        }}
    }}
    """

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "confidence": {"type": "number"},
                                "description": {"type": "string"},
                            },
                            "required": ["label", "confidence", "description"],
                        },
                    },
                    "summary": {"type": "string"},
                    "clinicalInsights": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "vitals": {
                        "type": "object",
                        "properties": {
                            "bpm": {"type": "number"},
                            "qrsDuration": {"type": "string"},
                            "qtInterval": {"type": "string"},
                        },
                        "required": ["bpm", "qrsDuration", "qtInterval"],
                    },
                },
                "required": ["predictions", "summary", "clinicalInsights", "vitals"],
            }
        )
    )
    return json.loads(response.text)

def main():
    st.set_page_config(page_title="CardiAI - ECG Analyzer", layout="wide")

    st.title("🫀 CardiAI - ECG Multi-Label Arrhythmia Analyzer")

    # Sidebar for file upload
    st.sidebar.header("Upload ECG Data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        # Initialize session state for files if not exists
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []

        # Process uploaded files
        if not st.session_state.processed_files:
            for file in uploaded_files:
                try:
                    df = parse_csv(file)
                    st.session_state.processed_files.append({
                        'name': file.name,
                        'data': df,
                        'status': 'idle',
                        'result': None
                    })
                except Exception as e:
                    st.sidebar.error(f"Error processing {file.name}: {str(e)}")


        # Display file status
        st.sidebar.subheader("File Status")
        for file_data in st.session_state.processed_files:
            status_icon = {
                'idle': '⏳',
                'analyzing': '🔄',
                'completed': '✅',
                'error': '❌'
            }.get(file_data['status'], '❓')

            if file_data['status'] == 'completed' and file_data['result']:
                prediction = file_data['result']['predictions'][0]['label'] if file_data['result']['predictions'] else 'Unknown'
                st.sidebar.write(f"{status_icon} {file_data['name']} - {prediction}")
            else:
                st.sidebar.write(f"{status_icon} {file_data['name']} - {file_data['status']}")

        # File selection
        file_names = [f['name'] for f in st.session_state.processed_files]
        selected_file = st.sidebar.selectbox("Select file to view", file_names)

        if selected_file:
            file_idx = file_names.index(selected_file)
            current_file = st.session_state.processed_files[file_idx]

            # Main content
            col1, col2 = st.columns([2, 1])

            with col1:
                st.header(f"📊 {current_file['name']}")

                # Lead selection
                active_leads = st.multiselect(
                    "Select leads to display",
                    LEADS,
                    default=['II', 'V1', 'V5'],
                    key=f"leads_{file_idx}"
                )

                if active_leads:
                    fig = plot_ecg(current_file['data'], active_leads)
                    st.plotly_chart(fig, use_container_width=True)

                # Show Clinical Summary and Insights right after ECG plot
                if current_file['status'] == 'completed' and current_file['result']:
                    st.markdown("---")  # Separator

                    # Clinical Summary
                    st.subheader("📋 Clinical Summary")
                    st.write(current_file['result']['summary'])

                    # Clinical Insights
                    st.subheader("💡 Clinical Insights")
                    for insight in current_file['result']['clinicalInsights']:
                        st.write(f"• {insight}")

            with col2:
                if current_file['status'] != 'completed':
                    if st.button("🔍 Analyze This File", type="primary", key=f"analyze_{file_idx}"):
                        with st.spinner("Analyzing with AI..."):
                            # Prepare sample data
                            sample_data = current_file['data'].head(100).to_json()
                            result = analyze_ecg_signal(sample_data, current_file['name'])
                            current_file['result'] = result
                            current_file['status'] = 'completed'
                            st.rerun()

                if current_file['status'] == 'completed' and current_file['result']:
                    display_analysis_results(current_file['result'])
                elif current_file['status'] == 'error':
                    st.error(f"Analysis failed: {current_file.get('error', 'Unknown error')}")

    else:
        st.info("👆 Upload CSV files containing ECG data to get started.")
        if 'processed_files' in st.session_state:
            del st.session_state.processed_files

def display_analysis_results(result: Dict[str, Any]):
    """Display analysis results in a dashboard format."""
    st.header("📈 Analysis Results")

    # Vitals
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Heart Rate", f"{result['vitals']['bpm']} BPM")
    with col2:
        st.metric("QRS Duration", result['vitals']['qrsDuration'])
    with col3:
        st.metric("QT Interval", result['vitals']['qtInterval'])

    # Predictions
    st.subheader("🔍 Detected Conditions")
    for pred in result['predictions']:
        with st.expander(f"{pred['label']} ({pred['confidence']*100:.1f}% confidence)"):
            st.write(pred['description'])

    # Detailed Probabilities Table
    st.subheader("📊 Detailed Probabilities")
    if result['predictions']:
        # Create DataFrame with all predictions sorted by confidence (highest first)
        prob_df = pd.DataFrame({
            'Class Name': [pred['label'] for pred in result['predictions']],
            'Probability (%)': [pred['confidence'] * 100 for pred in result['predictions']]
        }).sort_values('Probability (%)', ascending=False)

        # Display as styled table with red gradient
        st.dataframe(
            prob_df.style.background_gradient(
                cmap='Reds',
                subset=['Probability (%)']
            ).format({'Probability (%)': '{:.1f}'}),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()