#!/usr/bin/env python3
"""
Quick setuptools-scm test - run this before creating GitHub releases.
"""

import subprocess
import sys

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

def main():
    print("🔍 Quick setuptools-scm validation...\n")
    
    # Check Git status
    success, output, _ = run_cmd("git status --porcelain")
    if output.strip():
        print("❌ FAIL: Uncommitted changes detected")
        print("   Fix: Commit or stash changes before release")
        return 1
    else:
        print("✅ Git working directory is clean")
    
    # Check if at tag
    success, tag, _ = run_cmd("git describe --exact-match --tags HEAD")
    if success:
        print(f"✅ Currently at tag: {tag}")
    else:
        print("⚠️  Not at an exact tag - will generate dev version")
    
    # Check setuptools-scm version
    success, version, _ = run_cmd("python -m setuptools_scm")
    if success:
        print(f"✅ setuptools-scm version: {version}")
        if ".dev" in version:
            print("❌ FAIL: Development version detected")
            print("   This will be marked as pre-release on PyPI")
            return 1
    else:
        print("❌ FAIL: setuptools-scm not working")
        return 1
    
    print("\n🎉 Ready for GitHub release!")
    print(f"   Version {version} will be uploaded to PyPI as stable release")
    return 0

if __name__ == "__main__":
    sys.exit(main())
