#!/bin/bash
echo "Starting AI Orchestrator..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
