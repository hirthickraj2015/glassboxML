from setuptools import setup, find_packages

setup(
    name='glassboxml',
    version='0.1.0',
    description='A human-readable, auditable, and explainable ML model format.',
    author='Hirthick',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'shap',
        'numpy',
        'pandas',
        'matplotlib',
        'pyyaml'
    ],
    python_requires='>=3.8',
)
