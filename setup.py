"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='popmodel',
    version='0.0.1',
    description='Model molecular excited state populations over time',
    long_description=long_description,
    url='https://github.com/awbirdsall/popmodel',
    author='Adam Birdsall',
    author_email='abirdsall@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords=['spectroscopy', 'fluorescence', 'model', 'chemistry',
        'atmosphere'],
    package_dir = {'': 'src'},
    packages=['popmodel'],
    install_requires=['numpy','scipy','pyyaml'],
    package_data={
        'popmodel': ['data/parameters_sample.yaml'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'popmodel=popmodel.command_line:command',
        ],
    },
)
