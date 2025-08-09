#!/usr/bin/env python3
"""
Test runner script for Anomaly Spotter project.
Provides convenient interface for running different test categories.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description="Running command"):
    """Run shell command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Anomaly Spotter")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--file", "-f", help="Run specific test file")
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection
    if args.file:
        cmd.append(args.file)
    elif args.unit:
        cmd.append("tests/unit/")
    elif args.integration:
        cmd.append("tests/integration/")
    else:
        cmd.append("tests/")
    
    # Add markers
    markers = []
    if args.fast:
        markers.append("not slow")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    if not (project_root / "src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("🧪 Anomaly Spotter Test Runner")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Python: {sys.executable}")
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pytest not found. Installing...")
        if not run_command([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], 
                          "Installing pytest"):
            sys.exit(1)
    
    # Run the tests
    success = run_command(cmd, f"Running tests")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()