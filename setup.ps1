# setup.ps1
Write-Host "🚀 Setting up Dental AI System on Windows..." -ForegroundColor Green

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not installed. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Copy .env.example to .env if it doesn't exist
if (!(Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "✅ Created .env file - Please add your API keys!" -ForegroundColor Green
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Add your OpenAI API key to .env"
Write-Host "2. Run: docker-compose up"
Write-Host "3. Visit: http://localhost:8000/docs"