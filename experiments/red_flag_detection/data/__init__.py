# This file makes the directory a Python package
import sys
import os

# Add project root to path to enable absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root) 