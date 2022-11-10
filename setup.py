#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=8.0",
    "numpy>=1.22",
    "scipy>=1.7",
    "matplotlib>=3.5",
    "jupyterlab>=3.5",
]

setup_requirements = []

test_requirements = []

setup(
    author="luphord",
    author_email="luphord@protonmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A Python implementation of the Longstaff-Schwartz linear regression algorithm for the evaluation of call rights and American options.",
    entry_points={
        "console_scripts": [
            "longstaff_schwartz=longstaff_schwartz.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="longstaff_schwartz",
    name="longstaff_schwartz",
    packages=find_packages(include=["longstaff_schwartz"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/luphord/longstaff_schwartz",
    version="0.1.1",
    zip_safe=False,
)
