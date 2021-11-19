#!/usr/bin/env python3
import sys

if (sys.version_info[0] == 3 and sys.version_info[1] < 0):
    print("PSIMPL requires Python version 3.0 or later")
    sys.exit(1)

import os
import platform
import subprocess

from setuptools import setup, find_packages

DISTNAME = 'PSIMPL'
VERSION = '0.0.2' # psimpl.__version__
DESCRIPTION = 'Peptide-Spectrum match IMPutation Library'
with open('README.md') as f_in:
    LONG_DESCRIPTION = f_in.read()
AUTHOR = 'John T. Halloran'
AUTHOR_EMAIL = 'johnhalloran321@gmail.com'
URL = 'https://github.com/johnhalloran321/psimpl'
LICENSE='Apache 2.0'

CLASSIFIERS = ["Natural Language :: English",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Topic :: Scientific/Engineering :: Bio-Informatics",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3 :: Only"]

def main():
    setup(
        name=DISTNAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        packages=find_packages(include = ["psimpl"]),
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        url=URL,
        platforms=['any'],
        classifiers=CLASSIFIERS,
        entry_points = {'console_scripts': ['psimpl = psimpl.psimpl:main']}
    )

if __name__ == "__main__":
    main()
