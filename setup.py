from setuptools import setup, find_packages

setup(
    name="mathlogic",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'networkx>=2.5',
        'numpy>=1.20',
        'pandas>=1.2',
        'matplotlib>=3.3',
        'seaborn>=0.11',
        'numba>=0.53'
    ],
    author="HAL 9000",
    description="Mathematical Logic Analysis Framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7'
)

