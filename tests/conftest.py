# Ensure local 'wave' package takes precedence over stdlib
import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
