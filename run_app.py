import os
import webbrowser
import time

# Open browser automatically
time.sleep(2)
webbrowser.open("http://localhost:8501")

# Run Streamlit app
os.system("streamlit run app.py")