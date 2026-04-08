"""
conftest.py — pytest configuration for PortOps-LLM.
Adds the project root to sys.path so that `import env` works
regardless of the directory pytest is invoked from.
"""
import sys
import os

# Insert the project root (this file's directory) at the front of sys.path
sys.path.insert(0, os.path.dirname(__file__))
