from setuptools import setup


NAME = 'encodec'
DESCRIPTION = 'High fidelity neural audio codec'
URL = 'https://github.com/facebookresearch/encodec'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez, Jade Copet, Yossi Adi, Gabriel Synnaeve'
REQUIRES_PYTHON = '>=3.8.0'


setup(
    name=NAME,
    version='1.0',
    description=DESCRIPTION,
    long_description='',
    long_description_content_type='',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['encodec', 'encodec.quantization', 'encodec.modules'],
    # extras_require={
    #     'dev': ['flake8', 'mypy', 'pdoc3'],
    # },
    install_requires=['numpy', 'torch', 'torchaudio', 'einops'],
    include_package_data=True,
    # entry_points={
    #     'console_scripts': ['encodec=encodec.__main__:main'],
    # },
    license='Creative Commons Attribution-NonCommercial 4.0 International',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)