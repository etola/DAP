#!/usr/bin/env python
"""Minimal setup.py for backward compatibility with older pip/setuptools."""
from setuptools import setup

# Minimal metadata for legacy setup.py develop path.
# Full configuration is in pyproject.toml.
setup(
    name="ddpy-dap",
    version="0.1.0",
)
