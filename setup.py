#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='stochastic_approximation',
    version='0.1.0',
    description='Python implementation of various stochastic approximation algorithms',
    long_description=readme + '\n\n' + history,
    author='Huashuai Qu',
    author_email='quhuashuai@gmail.com',
    url='https://github.com/huashuai/stochastic_approximation',
    packages=[
        'stochastic_approximation',
    ],
    package_dir={'stochastic_approximation': 'stochastic_approximation'},
    include_package_data=True,
    install_requires=[
    ],
    license="BSD",
    zip_safe=False,
    keywords='stochastic_approximation',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
)