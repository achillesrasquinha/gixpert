

# imports - module imports
from gixpert.exception import (
    GixpertError
)

# imports - test imports
import pytest

def test_gixpert_error():
    with pytest.raises(GixpertError):
        raise GixpertError