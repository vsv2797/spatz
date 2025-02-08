#!/usr/bin/env python

# SpaTZ package for selection of scatterers for InSAR time series analysis
#
# Copyright (c) 2024  Andreas Piter (Institute of Photogrammetry and GeoInformation, piter@ipi.uni-hannover.de)
#
# This software was developed within the context [...]
#
# This program is not yet licensed and used for internal development only.

"""The setup script."""
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

version = {}
with open("spatz/version.py") as version_file:
    exec(version_file.read(), version)

req = []  # todo: add requirements

req_setup = []

req_test = ['pytest>=3', 'pytest-cov', 'pytest-reporter-html1', 'urlchecker']

req_doc = [
    'sphinx>=4.1.1',
    'sphinx-argparse',
    'sphinx-autodoc-typehints'
]

req_lint = ['flake8', 'pycodestyle', 'pydocstyle']

req_dev = ['twine'] + req_setup + req_test + req_doc + req_lint

setup(
    author="Andreas Piter",
    author_email='piter@ipi.uni-hannover.de',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: Release Candidate',
        'Intended Audience :: Researchers',
        'None',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="Selection of scatterers for InSAR time series analysis",
    entry_points={
        'console_scripts': [
            'spatz=spatz.spatz_cli:main',
            'spatz_plot=spatz.spatz_plot:main',
        ],
    },
    extras_require={
        "doc": req_doc,
        "test": req_test,
        "lint": req_lint,
        "dev": req_dev
    },
    install_requires=req,
    license="None",  # todo: add license
    include_package_data=True,
    keywords='spatz',
    long_description=readme,
    name='spatz',
    packages=find_packages(include=['spatz', 'spatz.*']),
    setup_requires=req_setup,
    test_suite='tests',
    tests_require=req_test,
    url='https://github.com/Andreas-Piter/spatz',
    version=version['__version__'],
    zip_safe=False,
)
