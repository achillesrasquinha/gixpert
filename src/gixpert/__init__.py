from __future__ import absolute_import

try:
    import os

    if os.environ.get("GIXPERT_GEVENT_PATCH"):
        from gevent import monkey
        monkey.patch_all(threaded = False, select = False)
except ImportError:
    pass

# imports - module imports
from gixpert.__attr__ import (
    __name__,
    __version__,
    __build__,

    __description__,

    __author__
)
from gixpert.__main__    import main
from gixpert.config      import PATH

from bpyutils.cache       import Cache
from bpyutils.config      import Settings, get_config_path
from bpyutils.util.jobs   import run_all as run_all_jobs, run_job

import deeply

cache = Cache(dirname = __name__)
cache.create()

settings = Settings(location = PATH["CACHE"], defaults = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "batch_norm": False,
    "dropout_rate": 0.5,
    "epochs": 56,
    "image_size": (512, 512)
})

def get_version_str():
    version = "%s%s" % (__version__, " (%s)" % __build__ if __build__ else "")
    return version

dops = deeply.ops.service("wandb")
dops.init("gixpert")