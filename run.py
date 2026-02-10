#!/usr/bin/env python3
"""
Run the Gas Town Orchestrator
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.server import run_server

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gas Town Orchestrator')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on')
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)
