import pytest
import effectful.ops.pyro
from pyroapi import pyro_backend


@pytest.fixture
def jit():
    return False


@pytest.fixture
def backend():
    with pyro_backend("effectful-minipyro"):
        yield


# noinspection PyUnresolvedReferences
from pyroapi.tests import *
