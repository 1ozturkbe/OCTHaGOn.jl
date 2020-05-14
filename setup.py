"""Standard Python setup script for OptimalConstraintTree"""
from setuptools import setup

LONG_DESCRIPTION = """
OptimalConstraintTree is a Python package for
generating optimization compatible constraints from data using
machine learning, and solving the resulting global
optimization problems.
"""

LICENSE = """ The MIT License

Copyright (c) 2020 Berk Ozturk <1ozturkbe@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

setup(
    name="OptimalConstraintTree",
    description="Package for generating optimization compatible "
                "constraints from data.",
    author="Berk Ozturk",
    author_email="bozturk@mit.edu",
    url="https://github.com/1ozturkbe/OptimalConstraintTree",
    install_requires=["gpkit >= 0.9", "interpretableai", "pyDOE", "progressbar",
                      "julia >= 0.5.0"],
    version="0.0.0",
    package_dir={"OptimalConstraintTree": "python"},
    packages=["OptimalConstraintTree",
              "OptimalConstraintTree.test"],
    long_description=LONG_DESCRIPTION,
)
