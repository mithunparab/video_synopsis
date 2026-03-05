"""Backward-compatible Streamlit entry point.

Usage:
    streamlit run app.py

For the new API, use:
    streamlit run video_synopsis/webui.py
"""

# Import and run the new WebUI module
import video_synopsis.webui  # noqa: F401
