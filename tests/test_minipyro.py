import pytest
from effectful.handlers.minipyro import default_runner
from pyroapi import pyro_backend

from effectful.ops.interpreter import interpreter


@pytest.fixture
def jit():
    return False


@pytest.fixture
def backend():
    with pyro_backend("effectful-minipyro"):
        with interpreter(default_runner):
            yield


# noinspection PyUnresolvedReferences
from pyroapi.tests import *
