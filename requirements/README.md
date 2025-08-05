# Requirements Structure

This directory contains organized dependency files for different use cases:

## Core Requirements
- **`main.txt`** - Main production dependencies (PyTorch, OpenCV, etc.)
- **`base.txt`** - Base core dependencies with pinned versions

## Environment-Specific Requirements  
- **`dev.txt`** - Development dependencies (includes base.txt + dev tools)
- **`prod.txt`** - Production-specific dependencies
- **`test.txt`** - Testing dependencies (pytest, coverage, etc.)
- **`docker.txt`** - Docker-specific requirements

## Feature-Specific Requirements
- **`wandb.txt`** - Weights & Biases integration and advanced features
- **`tools.txt`** - Development and analysis tools

## Usage Examples

### Development Setup
```bash
pip install -r requirements/dev.txt
```

### Production Setup  
```bash
pip install -r requirements/prod.txt
```

### Main Dependencies Only
```bash
pip install -r requirements/main.txt
```

### With W&B Features
```bash
pip install -r requirements/main.txt
pip install -r requirements/wandb.txt
```

### Using Root Requirements (Convenience)
```bash
# This automatically includes main.txt
pip install -r requirements.txt
```

## Notes
- The root `requirements.txt` is a convenience file that includes `main.txt`
- For conda environments, use `envs/environment.yml` instead
- Development dependencies automatically include base dependencies via `-r base.txt`
