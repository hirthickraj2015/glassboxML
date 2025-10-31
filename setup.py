from setuptools import setup, find_packages

setup(
    name='glassboxml',
    version='0.1.0',
    description='Human-readable machine learning model framework',
    author='Hirthick',
    packages=find_packages(),
    install_requires=['numpy'],
    python_requires='>=3.8',
)
