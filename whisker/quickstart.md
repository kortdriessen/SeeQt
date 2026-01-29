# Instructions for Viewing Texture Discrimination Exp:

First: make sure you have uv installed. Homebrew is one easy way on Mac. See here: https://docs.astral.sh/uv/getting-started/installation/ 

1. clone this entire repo using: git clone https://github.com/kortdriessen/SeeQt.git
2. In your terminal, navigate into the directory you just cloned (SeeQt)
3. Run: uv sync
   1. this will create a .venv directory and install the requisite python packages
4. Activate that ennvironment you just created, on mac: source .venv/bin/activate
5. go to the whisker directory: cd whisker
6. run: python -m streamlit run launch_whis.py
7. Follow instructions in that dashboard to specify a data directory, format and save the requisite information, and launch the viewer.