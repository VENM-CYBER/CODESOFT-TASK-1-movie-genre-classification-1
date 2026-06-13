# 🪟 Windows Setup Guide — CineGenre AI

## The Problem
You're running **Python 3.14**, which is too new for `scikit-learn==1.3.2`.
That version needs a C++ compiler to build from source on Windows.

The `requirements.txt` has been updated to use **flexible version ranges** so pip
automatically picks pre-built wheels — no compiler needed.

---

## ✅ Recommended: Use Python 3.11 or 3.12

Python 3.11 has the best pre-built wheel support for all ML packages.

### Step 1 — Install Python 3.11
Download from: https://www.python.org/downloads/release/python-3119/
- Choose **Windows installer (64-bit)**
- ✅ Check "Add Python to PATH" during install

Verify:
```cmd
python --version
# Should print: Python 3.11.x
```

### Step 2 — Create a virtual environment
```cmd
cd C:\Users\archi\OneDrive\Desktop\movie-genre-classification
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Step 3 — Install dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

This will now download **pre-built binary wheels** — no compiler required.

### Step 4 — Train the model
```cmd
python train.py
```

### Step 5 — Launch the app
```cmd
streamlit run streamlit_app\main.py
```

Open **http://localhost:8501** in your browser.

---

## ✅ Alternative: Keep Python 3.14, use `--only-binary`

If you want to keep Python 3.14, force pip to only use pre-built wheels:

```cmd
pip install -r requirements.txt --only-binary=:all:
```

If any package doesn't have a 3.14 wheel yet, you'll see a "no matching distribution"
error for that specific package. In that case, use Python 3.11 instead.

---

## ✅ Alternative: Use conda (no compiler ever needed)

```cmd
conda create -n cinegenre python=3.11
conda activate cinegenre
pip install -r requirements.txt
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `Microsoft Visual C++ 14.0 required` | Use Python 3.11 + `pip install -r requirements.txt` |
| `No module named 'streamlit'` | Run `pip install -r requirements.txt` inside the venv |
| `ModuleNotFoundError: src` | Make sure you `cd` into the project folder first |
| `FileNotFoundError: best_model.pkl` | Run `python train.py` before launching the app |
| Port 8501 already in use | Run `streamlit run streamlit_app\main.py --server.port 8502` |

---

## Full Command Sequence (copy-paste ready)

```cmd
:: 1. Open Command Prompt, navigate to project
cd C:\Users\archi\OneDrive\Desktop\movie-genre-classification

:: 2. Create & activate virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate

:: 3. Install all packages
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Train the model (generates data + saves artifacts)
python train.py

:: 5. Run all tests
python -m pytest tests\ -v

:: 6. Launch Streamlit app
streamlit run streamlit_app\main.py
```
