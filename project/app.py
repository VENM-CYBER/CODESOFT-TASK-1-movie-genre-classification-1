"""
app.py
-------
Entry-point shim – delegates to the Streamlit multi-page app.
Run with:  streamlit run app.py
"""

import streamlit.web.cli as stcli
import sys, os

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", os.path.join("streamlit_app", "main.py")]
    sys.exit(stcli.main())
