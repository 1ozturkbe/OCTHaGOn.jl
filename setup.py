"""Standard Python setup script for OptimalConstraintTree"""
import os
from distutils.core import setup

LONG_DESCRIPTION = """
OptimalConstraintTree is a Python package for 
generating optimization compatible constraints from data using 
machine learning. 
"""

setup(
    name="OptimalConstraintTree",
    description="Package for generating optimization compatible "
                "constraints from data.",
    author="Berk Ozturk",
    author_email="bozturk@mit.edu",
    url="https://github.com/1ozturkbe/OptimalConstraintTree",
    install_requires=["gpkit >= 0.9", "interpretableai", "pyDOE", "progressbar"],
    version="0.0.0",
    packages=["OptimalConstraintTree",
              "OptimalConstraintTree.testing"],
    long_description=LONG_DESCRIPTION,
)
