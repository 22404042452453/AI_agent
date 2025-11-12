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

# Start Ollama if not running
$ollamaProcess = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "Starting Ollama server..."
    Start-Process -NoNewWindow ollama serve
    Start-Sleep -Seconds 10  # Wait for Ollama to start
}

# Check and pull model if needed
Write-Host "Checking for model qwen3:8b..."
$modelExists = ollama list 2>$null | Select-String "qwen3:8b"
if (-not $modelExists) {
    Write-Host "Pulling model qwen3:8b... This may take some time."
    ollama pull qwen3:8b
}

# Start the app
Write-Host "Starting Streamlit app at http://localhost:8501/"
streamlit run app.py --server.address localhost
