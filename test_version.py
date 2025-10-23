#!/usr/bin/env python3
"""
Local testing script for setuptools-scm version generation.
Run this before creating GitHub releases to catch version issues early.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def test_setuptools_scm():
    """Test setuptools-scm version generation."""
    print(" Testing setuptools-scm version generation...")
    
    # Test 1: Check current version
    success, version, error = run_command("python -m setuptools_scm")
    if success:
        print(f" Current version: {version}")
        if ".dev" in version:
            print("  WARNING: Development version detected!")
            return False
    else:
        print(f" setuptools-scm failed: {error}")
        return False
    
    return True

def test_git_status():
    """Test Git repository status."""
    print("\n Checking Git repository status...")
    
    # Check for uncommitted changes
    success, output, error = run_command("git status --porcelain")
    if success:
        if output.strip():
            print("  WARNING: Uncommitted changes detected:")
            for line in output.strip().split('\n'):
                print(f"    {line}")
            return False
        else:
            print(" Working directory is clean")
    else:
        print(f" Git status check failed: {error}")
        return False
    
    return True

def test_git_tags():
    """Test Git tags availability."""
    print("\n Checking Git tags...")
    
    # Check if we're at a tag
    success, tag, error = run_command("git describe --exact-match --tags HEAD")
    if success:
        print(f" Currently at tag: {tag}")
    else:
        print("  Not at an exact tag")
        
        # Show latest tag
        success, latest_tag, error = run_command("git describe --tags --abbrev=0")
        if success:
            print(f"    Latest tag: {latest_tag}")
        
        # Show commits since latest tag
        success, commits, error = run_command("git rev-list --count HEAD ^$(git describe --tags --abbrev=0)")
        if success and commits:
            print(f"    Commits since latest tag: {commits}")
    
    return True

def test_clean_build():
    """Test building in a clean temporary directory (simulates GitHub Actions)."""
    print("\n Testing clean build (simulating GitHub Actions)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_repo = Path(temp_dir) / "repo"
        
        # Clone the current repository
        success, output, error = run_command(f"git clone . {temp_repo}")
        if not success:
            print(f" Failed to clone repository: {error}")
            return False
        
        # Test setuptools-scm in clean clone
        success, version, error = run_command("python -m setuptools_scm", cwd=temp_repo)
        if success:
            print(f" Clean clone version: {version}")
            if ".dev" in version:
                print("  WARNING: Clean clone still shows development version!")
                print("    This suggests the issue is with Git history or tags")
                return False
        else:
            print(f" setuptools-scm failed in clean clone: {error}")
            return False
    
    return True

def test_build_package():
    """Test package building in isolated environment."""
    print("\n Testing package build...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_repo = Path(temp_dir) / "repo"
        
        # Clone the current repository at the current tag
        success, current_tag, error = run_command("git describe --exact-match --tags HEAD")
        if success:
            success, output, error = run_command(f"git clone --branch {current_tag} . {temp_repo}")
        else:
            success, output, error = run_command(f"git clone . {temp_repo}")
        
        if not success:
            print(f" Failed to clone repository: {error}")
            return False
        
        # Build package in clean clone
        success, output, error = run_command("python -m build --wheel --no-isolation", cwd=temp_repo)
        if success:
            # Check what version was built
            success, files, error = run_command("ls dist/*.whl", cwd=temp_repo)
            if success:
                wheel_file = files.split('\n')[0] if files else ""
                print(f" Built wheel: {os.path.basename(wheel_file)}")
                
                # Extract version from filename
                if ".dev" in wheel_file:
                    print("  WARNING: Built package has development version!")
                    return False
            else:
                print(" No wheel file found")
                return False
        else:
            print(f" Package build failed: {error}")
            return False
    
    return True

def main():
    """Run all tests."""
    print(" Starting setuptools-scm local testing...\n")
    
    tests = [
        ("Git Status", test_git_status),
        ("Git Tags", test_git_tags),
        ("setuptools-scm Version", test_setuptools_scm),
        ("Clean Build Simulation", test_clean_build),
        ("Package Build", test_build_package),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f" {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print(" TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n All tests passed! Ready for GitHub release.")
        return 0
    else:
        print("\n  Some tests failed. Fix issues before creating GitHub release.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
