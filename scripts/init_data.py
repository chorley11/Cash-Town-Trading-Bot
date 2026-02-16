#!/usr/bin/env python3
"""
Initialize Data on Startup - Load seed data if no existing data

This script runs at startup to ensure trades_history.jsonl exists.
On Railway, the data directory is ephemeral, so we need to restore from:
1. Seed data in repo (data/seed/)
2. External backup (future: S3/Redis)

Usage:
    python scripts/init_data.py
"""
import os
import shutil
from pathlib import Path

# Paths
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get('DATA_DIR', REPO_DIR / 'data'))
SEED_DIR = REPO_DIR / 'data' / 'seed'

def init_data():
    """Initialize data directory with seed data if empty"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Files to initialize
    files = [
        'trades_history.jsonl',
        'strategy_performance.json',
    ]
    
    for filename in files:
        dest_file = DATA_DIR / filename
        seed_file = SEED_DIR / filename
        
        # Only copy if destination doesn't exist or is empty
        if not dest_file.exists() or dest_file.stat().st_size == 0:
            if seed_file.exists():
                print(f"📦 Initializing {filename} from seed data...")
                shutil.copy(seed_file, dest_file)
            else:
                print(f"⚠️ No seed data for {filename}")
        else:
            print(f"✅ {filename} already exists ({dest_file.stat().st_size} bytes)")

if __name__ == '__main__':
    init_data()
    print("✅ Data initialization complete")
