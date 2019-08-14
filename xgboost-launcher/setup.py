import logging
import os
import sys
import shutil

from setuptools import setup, find_packages

sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

logger = logging.getLogger(__name__)

file_path = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(
    os.path.join(file_path, 'launcher/version.py'),
    os.path.join(file_path, 'version.py'))
from version import get_xgboost_version, get_launcher_version

logger.info('\ninstall xgboost-launcher...\n')
setup(
    name='xgboost-launcher',
    version=get_launcher_version(),
    description="XGBoost Launcher Package",
    install_requires=[
        'ant-xgboost==%s' % get_xgboost_version(),
        'pandas==0.23.0',
        'pyyaml',
        'psutil',
        'oss2',
        'pyodps'
    ],
    maintainer='Xu Xiao',
    maintainer_email='lovedreamf@gmail.com',
    zip_safe=False,
    packages=find_packages(),
    license='Apache-2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Development Status :: 5 - Production/Stable',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],
    python_requires='>=3.6',
    url='https://github.com/alipay/ant-xgboost')

os.remove(os.path.join(file_path,'version.py'))
