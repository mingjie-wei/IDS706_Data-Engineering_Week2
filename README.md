# Ecommerce Consumer Behavior Analysis Project

## Project Overview


## Project Structure


## Development Container Setup

### Prerequisites
- Docker Desktop installed and running
- VS Code with Dev Containers extension

### Quick Start

#### 1. Clone the repository
```bash
git clone https://github.com/mingjie-wei/IDS706_Data-Engineering_Week2.git
cd IDS706_Data-Engineering_Week2
```

#### 2. Open in VS Code
```bash
code .
```

#### 3. Set up Dev Container
- Press Ctrl+Shift+P (Windows) or Shift+Command+P (Mac)

- Select "Dev Containers: Add Development Container Configuration Files"

- Choose "Add configuration to workspace" (recommended for team sharing)

- Select "Python" template → Choose Python version (e.g., 3.12)

- Skip additional features (optional)

- Optional: Add Dependabot configuration for automated dependency updates

#### 4. Reopen in Container
- Click "Reopen in Container" when prompted

- First-time build may take 5-15 minutes (depends on network speed)

### Verification
After successful build, confirm:

- VS Code status bar shows "Dev Container: Python 3"

- Terminal operates within the container environment

- Python packages are accessible

### Lessons Learned

#### 1. Successful Implementation
- Environment consistency: Dev Containers eliminate "works on my machine" issues

- Reproducibility: Containerization ensures identical environments across all setups

- Automation: Dependabot reduces maintenance overhead for dependency updates

#### 2. Challenges Overcome
- Initial setup: Required Docker and VS Code extension installation

- First build time: Significant initial download but fast subsequent builds

- Configuration choices: Selected workspace-level configuration for better collaboration

## Kaggle API Integration

### Prerequisites
- Kaggle account
- Kaggle API token

### Setup Steps

#### 1. Obtain Kaggle API Token
- Login to your Kaggle account

- Click on your profile picture → "Account"

- Scroll to "API" section

- Click "Create New API Token"

- This will download a kaggle.json file

#### 2. Secure Configuration
```bash
# Create directory for Kaggle configuration
mkdir kaggle

# Place kaggle.json in the directory (never commit this!)
# Then set up secure permissions within the development container
mkdir -p ~/.kaggle
cp kaggle/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # Critical security step
```

#### 3. Install Kaggle CLI
```bash
# Install within development environment
pip install kaggle
```

### Download Dataset
```bash
# Download the ecommerce dataset
kaggle datasets download -d salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data -p ./data

# Extract the compressed files
unzip ./data/ecommerce-consumer-behavior-analysis-data.zip -d ./data

# Alternative: Use Python for extraction
python -m zipfile -e ./data/ecommerce-consumer-behavior-analysis-data.zip ./data
```

### .gitignore Protection
Ensure your .gitignore contains:

```bash
# Kaggle API protection
kaggle/kaggle.json
*.json

# Data files protection
data/*.csv
data/*.zip
data/*.json
data/*/

# System files
.DS_Store
```

### Verification Commands
```bash
# Verify API configuration
kaggle --version

# Test API connectivity
kaggle datasets list -s "ecommerce" --max-size 3

# Verify file permissions (should show -rw-------)
ls -la ~/.kaggle/
```

## Jupyter Notebook Environment Setup

### Environment Configuration

#### 1. Create Notebooks Directory
```bash
# Create dedicated directory for Jupyter notebooks
mkdir notebooks

# Navigate to notebooks directory
cd notebooks

# Create main analysis notebook
touch ecommerce_behavior_analysis.ipynb

# Return to project root
cd ..
```

#### 2. Package Installation
```bash
# Install essential data science packages
pip install pandas matplotlib seaborn scikit-learn jupyter ipykernel

# Verify installation
python -c "import pandas as pd; print(f'pandas {pd.__version__} installed successfully')"
```

#### 3. Requirements Management
```bash
# Generate requirements.txt in project root
# pip freeze > requirements.txt
cat > requirements.txt <<'EOF'
pandas
matplotlib
seaborn
scikit-learn
jupyter
ipykernel
kaggle
EOF
```

### Jupyter Notebook Configuration

#### 1. Opening Notebooks
```bash
# Method 1: VS Code File Explorer
# Click on notebooks/ecommerce_behavior_analysis.ipynb

# Method 2: Terminal command
code notebooks/ecommerce_behavior_analysis.ipynb
```

#### 2. Kernel Selection
- Open .ipynb file in VS Code

- Click "Select Kernel" in top-right corner

- Choose: Python 3.12.11 (/usr/local/bin/python)

- Avoid system Python (/usr/bin/python3)

#### 3. Verification Cell
```bash
# Environment test cell
import sys
print("Python version:", sys.version)
print("Python path:", sys.executable)

# Test essential packages
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import datasets
    print("All data science packages available!")
    print(f"pandas version: {pd.__version__}")
except ImportError as e:
    print("Package import failed:", e)
```

