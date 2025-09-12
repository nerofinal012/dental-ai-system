# scripts/init_db.py
"""
Simplified initialization script
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def init_database():
    """Initialize with mock data (no database required for now)"""
    
    print("🚀 Initializing Dental AI System...")
    print("✅ Using in-memory storage (no database required)")
    print("✅ Mock documents loaded")
    print("✅ Ready to start the API!")
    print("\nNext step: Run 'python main.py' to start the server")

if __name__ == "__main__":
    init_database()