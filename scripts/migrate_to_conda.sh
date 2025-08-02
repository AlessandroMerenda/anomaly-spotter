#!/bin/bash

# Migration Guide: da venv a conda
# Esegui questo script per migrare dal virtual environment attuale a conda

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== MIGRATION DA VENV A CONDA ===${NC}"

# 1. Backup dell'ambiente attuale
echo -e "${YELLOW}1. Backup ambiente venv attuale...${NC}"
if [ -d "venv" ]; then
    echo "Trovato ambiente venv esistente"
    if [ -f "venv/bin/pip" ]; then
        echo "Esportando dipendenze attuali..."
        venv/bin/pip freeze > backup_requirements_venv.txt
        echo -e "${GREEN}✅ Backup salvato in backup_requirements_venv.txt${NC}"
    fi
else
    echo "Nessun ambiente venv trovato"
fi

# 2. Verifica conda
echo -e "${YELLOW}2. Verifica installazione conda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda non installato!${NC}"
    echo "Installa conda prima di continuare:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✅ Conda trovato: $(conda --version)${NC}"

# 3. Setup nuovo ambiente conda
echo -e "${YELLOW}3. Creazione ambiente conda...${NC}"
./scripts/setup_environment.sh

# 4. Test del nuovo ambiente
echo -e "${YELLOW}4. Test del nuovo ambiente...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate anomaly-spotter

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    print('✅ Import test passed')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

echo ""
echo -e "${GREEN}✅ Migration completata!${NC}"
echo ""
echo -e "${BLUE}Prossimi passi:${NC}"
echo -e "  1. ${GREEN}conda activate anomaly-spotter${NC}"
echo -e "  2. ${GREEN}python src/train_model.py${NC}"
echo ""
echo -e "${YELLOW}💡 Puoi rimuovere la vecchia cartella venv:${NC}"
echo -e "  ${RED}rm -rf venv/${NC}"
echo -e "  ${RED}rm -rf backup_requirements_venv.txt${NC}"
