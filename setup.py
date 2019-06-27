# Copyright 2019 The D'Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installs dsuite.

To install:
pip install -e .
"""

import os
import sys

import setuptools

MIN_PYTHON_VERSION = (3, 5, 3)

version = sys.version_info[:3]
if version < MIN_PYTHON_VERSION:
    print(('This package requires Python version at least {} (Current version '
           'is {})').format('.'.join(map(str, MIN_PYTHON_VERSION)), '.'.join(
               map(str, version))))
    sys.exit(1)

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_requirements(file_name):
    """Returns requirements from the given file."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


setuptools.setup(
    name="dsuite",
    version="0.1.0",
    license='Apache 2.0',
    description='Dexterous reinforcement learning environments.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=get_requirements('requirements.txt'),
    extra_requires={
        'dev': get_requirements('requirements.dev.txt'),
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
)
