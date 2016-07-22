from setuptools import setup

setup(name='ml_util',
      version='1.0.1',
      description='Library for machine learning',
      url='git@github.com:pauloalves86/ml_util.git',
      author='Paulo Alves',
      packages=['ml_util'],
      install_requires=['numpy', 'scikit-learn'],
      zip_safe=False)
