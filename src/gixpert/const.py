from bpyutils.const import CPU_COUNT
from bpyutils.util.environ import getenv

from gixpert import __name__ as NAME

_PREFIX = NAME.upper()

CONST = {
    "prefix": _PREFIX
}

DEFAULT = {
    "jobs":                 getenv("JOBS", CPU_COUNT, prefix = _PREFIX),
    "batch_size":           32,
    "learning_rate":        1e-5,
    "batch_norm":           False,
    "dropout_rate":         0.5,
    "epochs":               50,
    "image_width":          512,
    "image_height":         512
}