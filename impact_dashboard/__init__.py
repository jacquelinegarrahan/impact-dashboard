import configparser
import os

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

from . import _version
__version__ = _version.get_versions()['version']
