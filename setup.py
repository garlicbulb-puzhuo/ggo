from setuptools import setup
from setuptools import find_packages

setup(name='ggo',
      version='0.1',
      description='Deep learning on Spark with Keras',
      url='https://github.com/garlicbulb-puzhuo/ggo',
      author='Chen | Baiyu',
      install_requires=['elephas', 'keras', 'hyperas', 'flask'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
