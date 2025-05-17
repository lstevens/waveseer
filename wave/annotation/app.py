"""
Streamlit-based annotation interface for reviewing and correcting pattern detections.

This module provides a web interface for:
1. Visualizing detected patterns
2. Accepting/rejecting pattern detections
3. Adjusting pattern boundaries
4. Adding metadata and annotations
5. Submitting corrections for model improvement
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Import Waveseer modules
from wave.ml.viz.pattern_viz import PatternVisualizer

# Set page configuration
st.set_page_config(
    page_title="Waveseer Pattern Annotation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:9000")
DATA_DIR = Path(os.environ.get("WAVESEER_DATA_DIR", Path.home() / ".waveseer/data"))
FEEDBACK_FILE = DATA_DIR / "annotations.json"


class AnnotationApp:
    """Main application class for pattern annotation interface."""

    def __init__(self):
        """Initialize the annotation application."""
        self.visualizer = PatternVisualizer()
        self.setup_session_state()
        self.ensure_directories()

    def setup_session_state(self):
        """Initialize session state variables."""
        if "patterns" not in st.session_state:
            st.session_state.patterns = []
        if "current_pattern_idx" not in st.session_state:
            st.session_state.current_pattern_idx = 0
        if "annotations" not in st.session_state:
            st.session_state.annotations = []
        if "feedback_saved" not in st.session_state:
            st.session_state.feedback_saved = False
        if "api_connected" not in st.session_state:
            st.session_state.api_connected = False

    def ensure_directories(self):
        """Ensure required directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Create annotation file if it doesn't exist
        if not FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "w") as f:
                json.dump([], f)

    def check_api_connection(self) -> bool:
        """Check connection to the API server."""
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            st.session_state.api_connected = response.status_code == 200
            return st.session_state.api_connected
        except (requests.RequestException, requests.ConnectionError):
            st.session_state.api_connected = False
            return False

    def load_patterns(self) -> List[Dict]:
        """Load available patterns from the API."""
        try:
            response = requests.get(f"{API_URL}/catalog", timeout=5)
            if response.status_code == 200:
                patterns = response.json().get("patterns", [])
                st.session_state.patterns = patterns
                return patterns
            return []
        except (requests.RequestException, requests.ConnectionError):
            return []

    def detect_pattern(self, data: List[float], timeframe: str, use_ml: bool = True,
                     model_name: Optional[str] = None) -> Dict:
        """Detect patterns in the provided data using the API."""
        try:
            payload = {
                "tf": timeframe,
                "seq": data,
                "use_ml": use_ml
            }

            if model_name:
                payload["model_name"] = model_name

            response = requests.post(f"{API_URL}/match", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except (requests.RequestException, requests.ConnectionError):
            return {}

    def save_annotation(self,
                       pattern_id: str,
                       original_data: List[float],
                       user_feedback: Dict[str, Any]) -> bool:
        """Save user annotation to the feedback file."""
        try:
            # Read existing annotations
            annotations = []
            if FEEDBACK_FILE.exists():
                with open(FEEDBACK_FILE, "r") as f:
                    try:
                        annotations = json.load(f)
                    except json.JSONDecodeError:
                        annotations = []

            # Add new annotation
            annotation = {
                "pattern_id": pattern_id,
                "data": original_data,
                "feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            }

            annotations.append(annotation)

            # Write back to file
            with open(FEEDBACK_FILE, "w") as f:
                json.dump(annotations, f, indent=2)

            st.session_state.feedback_saved = True
            return True
        except Exception as e:
            st.error(f"Error saving annotation: {str(e)}")
            return False

    def plot_pattern(self, data: List[float], pattern_info: Dict, editable: bool = True) -> go.Figure:
        """Create an interactive plot of the pattern."""
        fig = go.Figure()

        # Add data trace
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode='lines',
            name='Price Data',
            line=dict(color='blue')
        ))

        # Get pattern information
        pattern_id = pattern_info.get("pattern_id", "")
        confidence = pattern_info.get("score", 0.5)
        pattern_type = pattern_info.get("pattern_type", "unknown")

        # Determine pattern color based on type
        color = self.visualizer.get_color_for_label(pattern_type)

        # Add pattern highlight
        # For simplicity, we'll highlight the entire range
        # In a more sophisticated implementation, we'd determine pattern start/end
        start_idx = 0
        end_idx = len(data) - 1

        fig.add_trace(go.Scatter(
            x=list(range(start_idx, end_idx + 1)),
            y=data[start_idx:end_idx + 1],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=confidence,
            name=f"Pattern: {pattern_type} ({confidence:.2f})",
            fill='tozeroy',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
        ))

        # Add verticals at pattern boundaries if editable
        if editable:
            fig.add_trace(go.Scatter(
                x=[start_idx, start_idx],
                y=[min(data), max(data)],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Start',
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=[end_idx, end_idx],
                y=[min(data), max(data)],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='End',
                hoverinfo='skip'
            ))

        # Configure layout
        fig.update_layout(
            title=f"Pattern: {pattern_type} (ID: {pattern_id})",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            dragmode='pan' if editable else None
        )

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='linear'
            )
        )

        return fig

    def run(self):
        """Run the Streamlit application."""
        st.title("🌊 Waveseer Pattern Annotation")

        # Sidebar
        with st.sidebar:
            st.header("Settings")

            # API connection status
            api_status = self.check_api_connection()
            st.write("API Status:",
                     "✅ Connected" if api_status else "❌ Disconnected")

            if not api_status:
                st.text_input("API URL", value=API_URL, key="api_url")
                if st.button("Connect"):
                    self.check_api_connection()

            # Load patterns
            if api_status and st.button("Load Patterns"):
                with st.spinner("Loading patterns..."):
                    patterns = self.load_patterns()
                    st.write(f"Loaded {len(patterns)} patterns")

            # Display annotation statistics
            st.header("Statistics")

            # Read annotations
            annotations = []
            if FEEDBACK_FILE.exists():
                with open(FEEDBACK_FILE, "r") as f:
                    try:
                        annotations = json.load(f)
                    except json.JSONDecodeError:
                        pass

            st.write(f"Total annotations: {len(annotations)}")

            # Count by feedback type
            if annotations:
                feedback_types = {}
                for annotation in annotations:
                    feedback = annotation.get("feedback", {})
                    feedback_type = feedback.get("type", "unknown")
                    feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1

                st.write("Feedback types:")
                for feedback_type, count in feedback_types.items():
                    st.write(f"- {feedback_type}: {count}")

        # Main content - Tabs
        tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Annotation History"])

        with tab1:
            st.header("Analyze Individual Patterns")

            # File upload or data entry
            data_source = st.radio("Data Source", ["Upload CSV", "Paste Data", "Generate Sample"])

            data = None
            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        column = st.selectbox("Select data column", df.columns)
                        data = df[column].values.tolist()
                        st.write(f"Loaded {len(data)} data points")
                    except Exception as e:
                        st.error(f"Error loading CSV: {str(e)}")

            elif data_source == "Paste Data":
                data_text = st.text_area("Paste data (one value per line or comma-separated)")
                if data_text:
                    try:
                        if "," in data_text:
                            data = [float(x.strip()) for x in data_text.split(",") if x.strip()]
                        else:
                            data = [float(x.strip()) for x in data_text.split("\n") if x.strip()]
                        st.write(f"Loaded {len(data)} data points")
                    except Exception as e:
                        st.error(f"Error parsing data: {str(e)}")

            elif data_source == "Generate Sample":
                pattern_type = st.selectbox("Sample Pattern Type",
                                           ["Head and Shoulders", "Double Top", "Double Bottom",
                                            "Triangle", "Random Walk", "Uptrend", "Downtrend"])
                length = st.slider("Pattern Length", 20, 200, 100)

                # Generate sample data based on pattern type
                if st.button("Generate"):
                    np.random.seed(int(time.time()))

                    if pattern_type == "Random Walk":
                        data = np.cumsum(np.random.normal(0, 1, length)).tolist()
                    elif pattern_type == "Uptrend":
                        noise = np.random.normal(0, 1, length)
                        data = (np.linspace(0, 10, length) + noise).tolist()
                    elif pattern_type == "Downtrend":
                        noise = np.random.normal(0, 1, length)
                        data = (np.linspace(10, 0, length) + noise).tolist()
                    elif pattern_type == "Head and Shoulders":
                        x = np.linspace(0, 6, length)
                        base = 3 * np.sin(x) + np.random.normal(0, 0.5, length)
                        # Add head and shoulders shape
                        peak_indices = [int(length/6), int(length/2), int(5*length/6)]
                        peak_heights = [2, 3, 2]
                        for idx, height in zip(peak_indices, peak_heights):
                            width = length / 10
                            for i in range(length):
                                dist = abs(i - idx)
                                if dist < width:
                                    base[i] += height * (1 - dist/width)
                        data = base.tolist()
                    else:
                        # Generate other patterns similarly
                        data = np.cumsum(np.random.normal(0, 1, length)).tolist()

                    st.write(f"Generated {len(data)} data points for {pattern_type}")

            # Detection controls
            if data and len(data) >= 10:
                col1, col2, col3 = st.columns(3)

                with col1:
                    use_ml = st.checkbox("Use ML Model", value=True)

                with col2:
                    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])

                with col3:
                    models = ["default"]
                    if st.session_state.api_connected:
                        try:
                            response = requests.get(f"{API_URL}/models", timeout=5)
                            if response.status_code == 200:
                                models_data = response.json()
                                models = ["default"] + [m["name"] for m in models_data]
                        except:
                            pass

                    model_name = st.selectbox("Model", models)
                    model_name = None if model_name == "default" else model_name

                # Detect button
                if st.button("Detect Patterns"):
                    with st.spinner("Detecting patterns..."):
                        pattern_result = self.detect_pattern(
                            data, timeframe, use_ml, model_name)

                        if pattern_result:
                            st.session_state.current_pattern = pattern_result
                            st.session_state.current_data = data

                            # Display results
                            st.subheader("Detection Results")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Pattern ID", pattern_result.get("pattern_id", "Unknown"))

                            with col2:
                                st.metric("Confidence", f"{pattern_result.get('score', 0) * 100:.1f}%")

                            with col3:
                                st.metric("Pattern Type", pattern_result.get("pattern_type", "Unknown"))

                            # Plot the pattern
                            fig = self.plot_pattern(data, pattern_result)
                            st.plotly_chart(fig, use_container_width=True)

                            # Feedback form
                            st.subheader("Provide Feedback")

                            feedback_type = st.radio("Is this detection correct?",
                                                   ["Correct", "Incorrect", "Partially Correct", "Unsure"])

                            corrected_type = st.text_input("Corrected Pattern Type (if applicable)")

                            comments = st.text_area("Additional Comments")

                            if st.button("Submit Feedback"):
                                feedback = {
                                    "type": feedback_type,
                                    "corrected_type": corrected_type if corrected_type else None,
                                    "comments": comments,
                                }

                                success = self.save_annotation(
                                    pattern_result.get("pattern_id", "unknown"),
                                    data,
                                    feedback
                                )

                                if success:
                                    st.success("Feedback submitted successfully!")
                                else:
                                    st.error("Error submitting feedback")
                        else:
                            st.error("No pattern detected or API error")

            elif data:
                st.warning("Data is too short. Please provide at least 10 data points.")

        with tab2:
            st.header("Batch Pattern Analysis")
            st.info("Upload multiple files or datasets for batch processing")

            # Batch upload
            uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

            if uploaded_files:
                batch_settings = st.expander("Batch Settings")

                with batch_settings:
                    col1, col2 = st.columns(2)

                    with col1:
                        batch_use_ml = st.checkbox("Use ML Model", value=True)
                        batch_timeframe = st.selectbox("Timeframe for All", ["1m", "5m", "15m", "1h", "4h", "1d"])

                    with col2:
                        batch_model_name = st.selectbox("Model for All", models)
                        batch_model_name = None if batch_model_name == "default" else batch_model_name

                        column_pattern = st.text_input("Column Name Pattern", "close")

                if st.button("Process Batch"):
                    total_files = len(uploaded_files)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    batch_results = []

                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})")

                            df = pd.read_csv(uploaded_file)

                            # Find the first column that matches the pattern
                            data_column = None
                            for col in df.columns:
                                if column_pattern.lower() in col.lower():
                                    data_column = col
                                    break

                            if data_column is None:
                                if 'close' in [c.lower() for c in df.columns]:
                                    data_column = df.columns[[c.lower() for c in df.columns].index('close')]
                                else:
                                    data_column = df.columns[0]

                            data = df[data_column].values.tolist()

                            if len(data) >= 10:
                                pattern_result = self.detect_pattern(
                                    data, batch_timeframe, batch_use_ml, batch_model_name)

                                batch_results.append({
                                    "filename": uploaded_file.name,
                                    "data": data,
                                    "result": pattern_result
                                })
                            else:
                                batch_results.append({
                                    "filename": uploaded_file.name,
                                    "data": data,
                                    "result": {"error": "Data too short"}
                                })
                        except Exception as e:
                            batch_results.append({
                                "filename": uploaded_file.name,
                                "error": str(e)
                            })

                        # Update progress
                        progress_bar.progress((i + 1) / total_files)

                    status_text.text(f"Processed {total_files} files")

                    # Display batch results
                    st.subheader("Batch Results")

                    for i, result in enumerate(batch_results):
                        with st.expander(f"{i+1}. {result['filename']}"):
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            elif "error" in result.get("result", {}):
                                st.warning(f"Warning: {result['result']['error']}")
                            else:
                                pattern_result = result["result"]
                                data = result["data"]

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Pattern ID", pattern_result.get("pattern_id", "Unknown"))

                                with col2:
                                    st.metric("Confidence", f"{pattern_result.get('score', 0) * 100:.1f}%")

                                with col3:
                                    st.metric("Pattern Type", pattern_result.get("pattern_type", "Unknown"))

                                # Plot the pattern
                                fig = self.plot_pattern(data, pattern_result, editable=False)
                                st.plotly_chart(fig, use_container_width=True)

                                # Add to annotation button
                                if st.button(f"Annotate {result['filename']}", key=f"annotate_{i}"):
                                    st.session_state.current_pattern = pattern_result
                                    st.session_state.current_data = data
                                    st.session_state.current_filename = result['filename']

                                    # Switch to the first tab
                                    # (not directly possible in current Streamlit, so we instruct the user)
                                    st.info("Pattern loaded. Please switch to the 'Single Analysis' tab to provide feedback.")

        with tab3:
            st.header("Annotation History")

            # Read annotations
            annotations = []
            if FEEDBACK_FILE.exists():
                with open(FEEDBACK_FILE, "r") as f:
                    try:
                        annotations = json.load(f)
                    except json.JSONDecodeError:
                        pass

            if annotations:
                # Filter and sort controls
                col1, col2 = st.columns(2)

                with col1:
                    sort_by = st.selectbox("Sort By", ["Timestamp (Newest First)",
                                                     "Timestamp (Oldest First)",
                                                     "Pattern ID",
                                                     "Feedback Type"])

                with col2:
                    filter_type = st.multiselect("Filter by Feedback Type",
                                               list(set(a.get("feedback", {}).get("type", "Unknown")
                                                     for a in annotations)),
                                               [])

                # Apply sorting
                if sort_by == "Timestamp (Newest First)":
                    annotations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                elif sort_by == "Timestamp (Oldest First)":
                    annotations.sort(key=lambda x: x.get("timestamp", ""))
                elif sort_by == "Pattern ID":
                    annotations.sort(key=lambda x: x.get("pattern_id", ""))
                elif sort_by == "Feedback Type":
                    annotations.sort(key=lambda x: x.get("feedback", {}).get("type", ""))

                # Apply filtering
                if filter_type:
                    annotations = [a for a in annotations
                                  if a.get("feedback", {}).get("type", "Unknown") in filter_type]

                # Display annotations
                for i, annotation in enumerate(annotations):
                    with st.expander(f"{i+1}. Pattern: {annotation.get('pattern_id', 'Unknown')} - {annotation.get('timestamp', 'Unknown')}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Pattern ID:**", annotation.get("pattern_id", "Unknown"))
                            st.write("**Timestamp:**", annotation.get("timestamp", "Unknown"))

                            feedback = annotation.get("feedback", {})
                            st.write("**Feedback Type:**", feedback.get("type", "Unknown"))

                            if feedback.get("corrected_type"):
                                st.write("**Corrected Type:**", feedback.get("corrected_type"))

                            if feedback.get("comments"):
                                st.write("**Comments:**", feedback.get("comments"))

                        with col2:
                            # Plot if data is available
                            if "data" in annotation:
                                data = annotation["data"]
                                pattern_info = {"pattern_id": annotation.get("pattern_id", "Unknown")}

                                if "feedback" in annotation and "corrected_type" in annotation["feedback"]:
                                    pattern_info["pattern_type"] = annotation["feedback"]["corrected_type"]

                                fig = self.plot_pattern(data, pattern_info, editable=False)
                                st.plotly_chart(fig, use_container_width=True)

                # Export annotations
                if st.button("Export Annotations"):
                    # Convert to CSV
                    csv_data = []
                    for annotation in annotations:
                        csv_row = {
                            "pattern_id": annotation.get("pattern_id", ""),
                            "timestamp": annotation.get("timestamp", ""),
                            "feedback_type": annotation.get("feedback", {}).get("type", ""),
                            "corrected_type": annotation.get("feedback", {}).get("corrected_type", ""),
                            "comments": annotation.get("feedback", {}).get("comments", ""),
                        }
                        csv_data.append(csv_row)

                    csv_df = pd.DataFrame(csv_data)
                    csv = csv_df.to_csv(index=False)

                    st.download_button(
                        "Download CSV",
                        csv,
                        "waveseer_annotations.csv",
                        "text/csv",
                        key="download-csv"
                    )
            else:
                st.info("No annotations found. Start analyzing patterns to create annotations.")


def main():
    """Run the annotation application."""
    app = AnnotationApp()
    app.run()


if __name__ == "__main__":
    main()
