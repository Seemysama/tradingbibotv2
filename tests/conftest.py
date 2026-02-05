"""
Pytest configuration.
Adds the project root to sys.path so that 'src' can be imported.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"Added to path: {project_root}")
