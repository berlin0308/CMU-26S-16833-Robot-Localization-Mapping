# SLAM Homework

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Prerequisites

Install `uv` if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Installing Dependencies

Install the project dependencies (this automatically creates a virtual environment and installs all dependencies):

```bash
uv sync
```

## Running the Code

After installing dependencies, run the particle filter code using `uv`:

```bash
uv run main.py
```

This automatically uses the project's virtual environment without needing to manually activate it.

Or activate the venv then run the code:

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\Scripts\activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Then run
python main.py
```