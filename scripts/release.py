#!/usr/bin/env python
"""
Release automation script for MuAI Orchestration System.

This script helps automate the release process including:
- Version validation
- Git tagging
- Build and packaging
- Release notes generation

Usage:
    python scripts/release.py --version 1.0.0-rc1 --dry-run
    python scripts/release.py --version 1.0.0-rc1 --execute
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.version import __version__, get_full_version, get_release_info


def run_command(cmd: str, dry_run: bool = False) -> tuple:
    """
    Run a shell command.
    
    Args:
        cmd: Command to run
        dry_run: If True, only print the command
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    print(f"{'[DRY RUN] ' if dry_run else ''}Running: {cmd}")
    
    if dry_run:
        return (0, "", "")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    return (result.returncode, result.stdout, result.stderr)


def validate_version(version: str) -> bool:
    """
    Validate version format.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Semantic versioning pattern with optional pre-release
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
    
    if not re.match(pattern, version):
        print(f"Error: Invalid version format: {version}")
        print("Expected format: X.Y.Z or X.Y.Z-prerelease")
        return False
    
    return True


def check_git_status(dry_run: bool = False) -> bool:
    """
    Check if git working directory is clean.
    
    Args:
        dry_run: If True, skip actual check
        
    Returns:
        True if clean, False otherwise
    """
    if dry_run:
        print("[DRY RUN] Checking git status...")
        return True
    
    returncode, stdout, _ = run_command("git status --porcelain")
    
    if returncode != 0:
        print("Error: Failed to check git status")
        return False
    
    if stdout.strip():
        print("Error: Git working directory is not clean")
        print("Please commit or stash your changes first")
        print(stdout)
        return False
    
    print("✓ Git working directory is clean")
    return True


def create_git_tag(version: str, dry_run: bool = False) -> bool:
    """
    Create and push git tag.
    
    Args:
        version: Version to tag
        dry_run: If True, only print commands
        
    Returns:
        True if successful, False otherwise
    """
    tag_name = f"v{version}"
    message = f"Release {version}"
    
    # Create tag
    returncode, _, _ = run_command(
        f'git tag -a {tag_name} -m "{message}"',
        dry_run
    )
    
    if returncode != 0 and not dry_run:
        print(f"Error: Failed to create tag {tag_name}")
        return False
    
    print(f"✓ Created tag {tag_name}")
    
    # Push tag
    returncode, _, _ = run_command(
        f"git push origin {tag_name}",
        dry_run
    )
    
    if returncode != 0 and not dry_run:
        print(f"Error: Failed to push tag {tag_name}")
        return False
    
    print(f"✓ Pushed tag {tag_name}")
    return True


def build_package(dry_run: bool = False) -> bool:
    """
    Build distribution packages.
    
    Args:
        dry_run: If True, only print commands
        
    Returns:
        True if successful, False otherwise
    """
    # Clean old builds
    returncode, _, _ = run_command(
        "rm -rf dist/ build/ *.egg-info",
        dry_run
    )
    
    if returncode != 0 and not dry_run:
        print("Error: Failed to clean old builds")
        return False
    
    print("✓ Cleaned old builds")
    
    # Build packages
    returncode, _, _ = run_command(
        "python setup.py sdist bdist_wheel",
        dry_run
    )
    
    if returncode != 0 and not dry_run:
        print("Error: Failed to build packages")
        return False
    
    print("✓ Built distribution packages")
    return True


def verify_package(dry_run: bool = False) -> bool:
    """
    Verify built packages.
    
    Args:
        dry_run: If True, only print commands
        
    Returns:
        True if successful, False otherwise
    """
    returncode, _, _ = run_command(
        "twine check dist/*",
        dry_run
    )
    
    if returncode != 0 and not dry_run:
        print("Error: Package verification failed")
        return False
    
    print("✓ Package verification passed")
    return True


def print_release_info():
    """Print release information."""
    info = get_release_info()
    
    print("\n" + "=" * 60)
    print("Release Information")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Release Name: {info['release_name']}")
    print(f"Release Date: {info['release_date']}")
    print(f"Release Status: {info['release_status']}")
    print(f"Build Number: {info['build_number']}")
    print(f"Full Version: {get_full_version()}")
    print("\nEnabled Features:")
    for feature, enabled in info['features'].items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")
    print("=" * 60 + "\n")


def print_next_steps(version: str):
    """Print next steps after release."""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print(f"1. Create GitHub Release for v{version}")
    print("   - Go to: https://github.com/your-org/muai-orchestration/releases/new")
    print(f"   - Tag: v{version}")
    print(f"   - Title: v{version}")
    print("   - Description: Copy from RELEASE_NOTES_v{version}.md")
    print("   - Mark as pre-release if RC/beta/alpha")
    print()
    print("2. Upload distribution files to GitHub Release")
    print("   - dist/*.tar.gz")
    print("   - dist/*.whl")
    print()
    print("3. (Optional) Publish to PyPI")
    print("   - Test: twine upload --repository testpypi dist/*")
    print("   - Prod: twine upload dist/*")
    print()
    print("4. Announce the release")
    print("   - Update project website")
    print("   - Post on social media")
    print("   - Notify users via email")
    print("=" * 60 + "\n")


def main():
    """Main release function."""
    parser = argparse.ArgumentParser(
        description="Release automation for MuAI Orchestration System"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version to release (e.g., 1.0.0-rc1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without making changes"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the release (required for actual release)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Print current version info
    print_release_info()
    
    # Determine version
    if args.version:
        version = args.version
    else:
        version = __version__
        print(f"Using current version: {version}")
    
    # Validate version
    if not validate_version(version):
        sys.exit(1)
    
    # Check if version matches current
    if version != __version__:
        print(f"Warning: Specified version ({version}) differs from current version ({__version__})")
        print("Please update mm_orch/version.py first")
        sys.exit(1)
    
    # Determine if dry run
    dry_run = args.dry_run or not args.execute
    
    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("EXECUTING RELEASE - Changes will be made!")
        print("=" * 60 + "\n")
        
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != "yes":
            print("Release cancelled")
            sys.exit(0)
    
    # Step 1: Check git status
    print("\n[Step 1/5] Checking git status...")
    if not check_git_status(dry_run):
        sys.exit(1)
    
    # Step 2: Run tests (optional)
    if not args.skip_tests:
        print("\n[Step 2/5] Running tests...")
        returncode, _, _ = run_command("pytest tests/ -v", dry_run)
        if returncode != 0 and not dry_run:
            print("Error: Tests failed")
            sys.exit(1)
        print("✓ Tests passed")
    else:
        print("\n[Step 2/5] Skipping tests (--skip-tests)")
    
    # Step 3: Create git tag
    print("\n[Step 3/5] Creating git tag...")
    if not create_git_tag(version, dry_run):
        sys.exit(1)
    
    # Step 4: Build package
    print("\n[Step 4/5] Building distribution packages...")
    if not build_package(dry_run):
        sys.exit(1)
    
    # Step 5: Verify package
    print("\n[Step 5/5] Verifying packages...")
    if not verify_package(dry_run):
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("✓ Release process completed successfully!")
    print("=" * 60)
    
    # Print next steps
    print_next_steps(version)


if __name__ == "__main__":
    main()
