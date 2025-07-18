# Running Sri Lankan Legal AI Assistant in Google Colab

This guide will help you run both the Vector_DB_module and RAG_Module in Google Colab.

## 📋 Prerequisites

1. Google account with access to Google Colab
2. Your project files uploaded to Google Drive or GitHub
3. Basic familiarity with Jupyter notebooks

## 🚀 Method 1: Upload Files to Google Drive

### Step 1: Prepare Your Files
1. Zip your entire project folder: `Vector_DB_module and RGA_Module`
2. Upload the zip file to your Google Drive

### Step 2: Create a New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Add the following cells:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Install Required Packages
!pip install streamlit
!pip install pinecone-client
!pip install sentence-transformers
!pip install transformers
!pip install torch
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install nltk
!pip install spacy
!pip install fastapi
!pip install uvicorn
!pip install python-multipart
!pip install pydantic
!pip install python-dotenv
!pip install plotly
!pip install wordcloud
!pip install matplotlib
!pip install seaborn
```

```python
# Cell 3: Extract and Setup Project
import zipfile
import os

# Change this path to where you uploaded your zip file
zip_path = '/content/drive/MyDrive/Vector_DB_module and RGA_Module.zip'

# Extract the project
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Change to project directory
os.chdir('/content/Vector_DB_module and RGA_Module')
!ls -la
```

```python
# Cell 4: Setup Environment Variables
import os

# Set up environment variables (replace with your actual values)
os.environ['PINECONE_API_KEY'] = 'your-pinecone-api-key-here'
os.environ['PINECONE_ENVIRONMENT'] = 'your-pinecone-environment-here'
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'  # Optional

# Create .env file
with open('.env', 'w') as f:
    f.write(f"PINECONE_API_KEY={os.environ['PINECONE_API_KEY']}\n")
    f.write(f"PINECONE_ENVIRONMENT={os.environ['PINECONE_ENVIRONMENT']}\n")
    f.write(f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}\n")
```

## 🗄️ Method 2: Clone from GitHub (Recommended)

### Step 1: Upload to GitHub
1. Create a GitHub repository
2. Upload your project files to the repository

### Step 2: Clone in Colab
```python
# Cell 1: Clone Repository
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name
```

```python
# Cell 2: Install Requirements
!pip install -r requirements.txt
# Or install packages individually as shown in Method 1
```

## 🏃‍♂️ Running the Modules

### Option A: Run Vector DB Module First

```python
# Cell: Run Vector DB Module
%cd Vector_DB_module
!python main.py
```

### Option B: Run RAG Module with Web Interface

```python
# Cell: Install Streamlit tunnel for Colab
!pip install pyngrok

# Set up ngrok for public URL
from pyngrok import ngrok
import threading
import time

# Start ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")
```

```python
# Cell: Run RAG Module Web Interface
%cd RAG_Module
!streamlit run frontend.py --server.port 8501 --server.headless true
```

### Option C: Run RAG Module CLI

```python
# Cell: Run RAG Module CLI
%cd RAG_Module
!python app.py cli
```

## 🔧 Colab-Specific Configurations

### Handle File Paths
```python
# Cell: Fix file paths for Colab
import os
import sys

# Add project root to Python path
project_root = '/content/Vector_DB_module and RGA_Module'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Vector_DB_module'))
sys.path.append(os.path.join(project_root, 'RAG_Module'))
```

### Memory Management
```python
# Cell: Check GPU and RAM
!nvidia-smi
!cat /proc/meminfo | grep MemTotal
```

### Persistent Storage
```python
# Cell: Save results to Drive
import shutil

# Save outputs to Google Drive
output_dir = '/content/drive/MyDrive/Legal_AI_Results'
os.makedirs(output_dir, exist_ok=True)

# Copy results
if os.path.exists('results'):
    shutil.copytree('results', f'{output_dir}/results', dirs_exist_ok=True)
```

## 🌐 Web Interface in Colab

### Using ngrok for Public Access
```python
# Cell: Setup ngrok
!pip install pyngrok
from pyngrok import ngrok
import subprocess
import threading
import time

# Kill any existing streamlit processes
!pkill -f streamlit

# Start Streamlit in background
def run_streamlit():
    subprocess.run(['streamlit', 'run', 'RAG_Module/frontend.py', 
                   '--server.port', '8501', '--server.headless', 'true'])

# Start streamlit in a separate thread
streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.daemon = True
streamlit_thread.start()

# Wait a moment for streamlit to start
time.sleep(10)

# Create ngrok tunnel
public_url = ngrok.connect(8501)
print(f"🌐 Access your Legal AI Assistant at: {public_url}")
```

## 📊 Complete Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# === CELL 1: Setup ===
from google.colab import drive
import os
import sys

# Mount Google Drive
drive.mount('/content/drive')

# === CELL 2: Install Packages ===
!pip install streamlit pinecone-client sentence-transformers transformers torch
!pip install numpy pandas scikit-learn nltk spacy fastapi uvicorn
!pip install python-multipart pydantic python-dotenv plotly pyngrok

# === CELL 3: Get Project Files ===
# Option A: From Drive
!unzip "/content/drive/MyDrive/Vector_DB_module and RGA_Module.zip" -d /content/

# Option B: From GitHub
# !git clone https://github.com/yourusername/your-repo.git /content/legal-ai

# === CELL 4: Setup Environment ===
os.chdir('/content/Vector_DB_module and RGA_Module')
sys.path.append('/content/Vector_DB_module and RGA_Module')

# Set environment variables
os.environ['PINECONE_API_KEY'] = 'your-key-here'
os.environ['PINECONE_ENVIRONMENT'] = 'your-env-here'

# === CELL 5: Run Web Interface ===
from pyngrok import ngrok
import subprocess
import threading
import time

# Start Streamlit
def run_app():
    os.chdir('/content/Vector_DB_module and RGA_Module/RAG_Module')
    subprocess.run(['python', 'app.py', 'web'])

thread = threading.Thread(target=run_app)
thread.daemon = True
thread.start()

time.sleep(15)  # Wait for app to start

# Create public URL
public_url = ngrok.connect(8501)
print(f"🚀 Your Legal AI Assistant is ready!")
print(f"🌐 Access it at: {public_url}")
```

## 🔑 Important Notes

1. **API Keys**: Make sure to set your Pinecone API key
2. **Memory**: Colab has limited RAM, monitor usage
3. **Session Timeout**: Colab sessions timeout after inactivity
4. **File Persistence**: Save important results to Google Drive
5. **ngrok**: Free tier has limitations, consider upgrading for heavy use

## 🐛 Troubleshooting

### Common Issues:
- **Import Errors**: Add project paths to sys.path
- **Port Issues**: Use different ports if 8501 is busy
- **Memory Errors**: Restart runtime and clear outputs
- **Timeout**: Keep the browser tab active

### Quick Fixes:
```python
# Restart everything
!pkill -f streamlit
!pkill -f python
# Then re-run the setup cells
```

This setup will allow you to run your Sri Lankan Legal AI Assistant in Google Colab with full web interface access! 
