#!/bin/bash
# Anomaly Spotter - Dependency Management Script
# Gestisce installazione e aggiornamento delle dipendenze per diversi ambienti

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a virtual environment
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Non sei in un virtual environment!"
        print_warning "Consigliato: python -m venv venv && source venv/bin/activate"
        read -p "Continuare comunque? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_status "Virtual environment attivo: $VIRTUAL_ENV"
    fi
}

# Function to install dependencies for specific environment
install_deps() {
    local env=$1
    local requirements_file=""
    
    case $env in
        "dev"|"development")
            requirements_file="requirements-dev.txt"
            print_status "Installazione dipendenze per sviluppo..."
            ;;
        "test"|"testing")
            requirements_file="requirements-test.txt"
            print_status "Installazione dipendenze per testing..."
            ;;
        "prod"|"production")
            requirements_file="requirements-prod.txt"
            print_status "Installazione dipendenze per produzione..."
            ;;
        "docker")
            requirements_file="requirements-docker.txt"
            print_status "Installazione dipendenze per Docker..."
            ;;
        "tools")
            requirements_file="requirements-tools.txt"
            print_status "Installazione tools per gestione dipendenze..."
            ;;
        *)
            requirements_file="requirements.txt"
            print_status "Installazione dipendenze base..."
            ;;
    esac
    
    if [[ ! -f "$requirements_file" ]]; then
        print_error "File $requirements_file non trovato!"
        exit 1
    fi
    
    pip install --upgrade pip
    pip install -r "$requirements_file"
    print_success "Dipendenze installate da $requirements_file"
}

# Function to check for security vulnerabilities
security_check() {
    print_status "Controllo vulnerabilità di sicurezza..."
    
    if ! command -v safety &> /dev/null; then
        print_warning "Safety non installato. Installazione in corso..."
        pip install safety
    fi
    
    safety check
    print_success "Controllo sicurezza completato"
}

# Function to audit dependencies
audit_deps() {
    print_status "Audit delle dipendenze..."
    
    if ! command -v pip-audit &> /dev/null; then
        print_warning "pip-audit non installato. Installazione in corso..."
        pip install pip-audit
    fi
    
    pip-audit
    print_success "Audit completato"
}

# Function to show dependency tree
show_tree() {
    print_status "Visualizzazione albero delle dipendenze..."
    
    if ! command -v pipdeptree &> /dev/null; then
        print_warning "pipdeptree non installato. Installazione in corso..."
        pip install pipdeptree
    fi
    
    pipdeptree
}

# Function to check for outdated packages
check_outdated() {
    print_status "Controllo pacchetti obsoleti..."
    pip list --outdated --format=columns
}

# Function to show help
show_help() {
    cat << EOF
Anomaly Spotter - Dependency Management Script

Uso: $0 [COMMAND] [ENVIRONMENT]

COMMANDS:
    install [ENV]    Installa dipendenze per ambiente specifico
                     ENV: dev|test|prod|docker|tools (default: base)
    
    security         Controlla vulnerabilità di sicurezza
    audit           Esegue audit delle dipendenze
    tree            Mostra albero delle dipendenze
    outdated        Mostra pacchetti obsoleti
    help            Mostra questo help

ESEMPI:
    $0 install dev          # Installa dipendenze per sviluppo
    $0 install prod         # Installa dipendenze per produzione
    $0 security             # Controllo sicurezza
    $0 audit                # Audit dipendenze
    $0 tree                 # Visualizza albero dipendenze

EOF
}

# Main script logic
case "${1:-help}" in
    "install")
        check_venv
        install_deps "${2:-base}"
        ;;
    "security")
        security_check
        ;;
    "audit")
        audit_deps
        ;;
    "tree")
        show_tree
        ;;
    "outdated")
        check_outdated
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Comando sconosciuto: $1"
        show_help
        exit 1
        ;;
esac
