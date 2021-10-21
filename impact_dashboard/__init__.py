import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

from . import _version
__version__ = _version.get_versions()['version']
