
"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
    name='spatiospectral_diarization',
    version='0.0.0',
    description='Spatio-spectral diarization of meetings by combining TDOA-based segmentation and speaker embedding-based clustering',  # Optional

    long_description=long_description,

    long_description_content_type='text/markdown',
    url='https://github.com/fgnt/spatiospectral_diarization',

    author='Tobias Cord-Landwehr, Tobias Gburrek, Marc Deegen',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

    ],

    keywords='diarization spatio-spectral multichannel',  # Optional


    packages=find_packages(exclude=['notebooks']),  # Required


    python_requires='>=3.8, <4',


    install_requires=[
        'numpy',
        'scikit_learn',
        'einops',
        'dlp_mpi',
        'paderbox',
        'padertorch',
        'meeteval',
        'nara_wpe',
    ],  # Optional


    extras_require={  # Optional
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },

    package_data={  # Optional
        # 'sample': ['package_data.dat'],
    },

    data_files=[
        # ('my_data', ['data/data_file'])
    ],  # Optional
)
