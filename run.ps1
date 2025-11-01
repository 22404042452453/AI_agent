Write-Host "Starting AI Agent..."

# Create directories if needed
if (-not (Test-Path "files")) { New-Item -ItemType Directory -Path "files" -Force }
if (-not (Test-Path "files_TT")) { New-Item -ItemType Directory -Path "files_TT" -Force }
if (-not (Test-Path "faiss_index")) { New-Item -ItemType Directory -Path "faiss_index" -Force }

# Create indexes if needed
if ((-not (Test-Path "faiss_index/index.faiss")) -or (-not (Test-Path "faiss_index/index.pkl"))) {
    Write-Host "Creating document indexes..."
    Write-Host "Note: This may take several minutes for first-time setup..." -ForegroundColor Cyan
    python main.py --create-indexes
}

# Start the app
Write-Host "Starting Streamlit app at http://localhost:8501"
streamlit run app.py
