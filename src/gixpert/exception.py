# imports - standard imports
import subprocess as sp

class GixpertError(Exception):
    pass

class PopenError(GixpertError, sp.CalledProcessError):
    pass

class DependencyNotFoundError(ImportError):
    pass