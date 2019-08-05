import logging
import os
import subprocess
import sys
import shutil

from pip._internal import main as pip_main
from setuptools import setup, find_packages

sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

LAUCNHER_VERSION = open(os.path.join(CURRENT_DIR, 'launcher/VERSION')).read().strip()

REQUIRED_XGBOOST_VERSION = open(
    os.path.join(CURRENT_DIR, '../python-package/xgboost/VERSION')).read().strip()

logger = logging.getLogger(__name__)


def install_xgboost():
    logger.info('\ninstall latest Ant-XGBoost from source code...\n')
    os.chdir('..')
    if os.path.exists('build'):
        shutil.rmtree('build')
    os.mkdir('build')
    os.chdir('build')
    if subprocess.run(['cmake', '..']).returncode or subprocess.run(['make', '-j4']).returncode:
        raise EnvironmentError('Failed to build xgboost library!')
    os.chdir('../python-package')
    if subprocess.run(['python3', 'setup.py', 'install']).returncode:
        raise EnvironmentError('Failed to install xgboost python package!')
    shutil.rmtree('../build')
    os.chdir('../xgboost-launcher')


def check_and_install_xgboost():
    try:
        import xgboost
        assert xgboost.__version__ == REQUIRED_XGBOOST_VERSION
        from xgboost import automl_core
        logger.info('Latest Ant-XGBoost has already been installed! Install xgblauncher directly!\n')
    except Exception as e:
        try:
            import xgboost
            logger.warning('\nFound incompatible xgboost version, try to remove it...\n')
            if pip_main(['show', 'xgboost']) != 0:
                raise EnvironmentError(
                    'xgboost of dismatch version detected, which is not installed via pip. '
                    'Please uninstall it manually!')
            pip_main(['uninstall', 'xgboost', '-y'])
        except Exception as e:
            pass
        finally:
            install_xgboost()


check_and_install_xgboost()
logger.info('\ninstall xgblauncher...\n')
setup(
    name='xgblauncher',
    version=LAUCNHER_VERSION,
    description="XGBoost Launcher Package",
    install_requires=[
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
