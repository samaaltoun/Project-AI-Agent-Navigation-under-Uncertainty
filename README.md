# AI Agent Navigation under Uncertainty

This project uses **Python 3.12** and **Gymnasium**.

Please follow the steps below carefully to set up your environment.

---

# 🔧 1. Install Git

### Mac
Open Terminal and run:
```bash
git --version
```

If not installed:
```bash
xcode-select --install
```

### Windows
Download Git from:
https://git-scm.com/

After installing, verify:
```powershell
git --version
```

---

# 📥 2. Clone the Repository

```bash
git clone <REPO_URL>
cd Project-AI-Agent-Navigation-under-Uncertainty
```

Make sure you are inside the project folder before continuing.

---

# 🐍 3. Install Python 3.12

This project requires **Python 3.12**.

Check your version:

```bash
python3 --version
```

It must show:

```
Python 3.12.x
```

If not installed, download it from:
https://www.python.org/downloads/

⚠ Windows users: During installation, check **"Add Python to PATH"**

---

# 🧪 4. Create Virtual Environment

Inside the project folder:

```bash
python3 -m venv .venv
```

---

# ▶ 5. Activate Virtual Environment

### Mac
```bash
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

You should now see:

```
(.venv)
```

Verify:

```bash
python --version
```

It should show:
```
Python 3.12.x
```

---

# 📦 6. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ✅ 7. Smoke Test (Verify Setup)

Run this test:

```bash
python -c "import gymnasium as gym; env=gym.make('CartPole-v1'); obs,info=env.reset(); print('Observation shape:', obs.shape); env.close(); print('SMOKE TEST PASSED ✅')"
```

Expected output:

```
Observation shape: (4,)
SMOKE TEST PASSED ✅
```

If you see this, your setup is correct.

---

# 🖥 Optional: Rendering Test

```bash
python src/render_test.py
```

If a CartPole window opens and runs, rendering works.

If rendering fails, install:

```bash
pip install pygame
```

---

# 🔄 Daily Workflow

Every time you work on the project:

```bash
cd Project-AI-Agent-Navigation-under-Uncertainty
source .venv/bin/activate   # Mac
```

or

```powershell
.\.venv\Scripts\Activate.ps1   # Windows
```

---

# ⚠ Important Rules

- DO NOT commit `.venv/`
- Always activate the virtual environment before running code
- Do not install packages globally

If someone installs a new package:

```bash
pip install <package>
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

Then everyone else must run:

```bash
pip install -r requirements.txt
```

---

# 🚀 Setup Complete

If everything runs without errors, you are ready to start working on the project.
