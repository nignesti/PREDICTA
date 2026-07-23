import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, "pages")

for path in (ROOT, PAGES):
    if path not in sys.path:
        sys.path.insert(0, path)
