"""
Command-line interface for DeepCausalMMM.

This module provides a CLI for training and using the model.
"""

import argparse
import json
import sys
from deepcausalmmm.core.config import get_default_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DeepCausalMMM: Deep Learning Marketing Mix Model with Causal Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show configuration
  deepcausalmmm config
  
  # Show version
  deepcausalmmm version
  
  # For training and analysis, use the Python API:
  # from deepcausalmmm import DeepCausalMMM, ModelTrainer
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show default configuration')
    
    # Version command  
    version_parser = subparsers.add_parser('version', help='Show package version')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'config':
            config_command(args)
        elif args.command == 'version':
            version_command(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def config_command(args):
    """
    Show default configuration.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """
    config = get_default_config()
    print("DeepCausalMMM Default Configuration:")
    print("=" * 40)
    print(json.dumps(config, indent=2))


def version_command(args):
    """
    Show package version.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """
    try:
        from deepcausalmmm import __version__
        print(f"DeepCausalMMM version {__version__}")
    except ImportError:
        print("DeepCausalMMM version 1.0.0")


if __name__ == "__main__":
    main()
