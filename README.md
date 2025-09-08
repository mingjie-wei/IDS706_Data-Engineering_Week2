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
