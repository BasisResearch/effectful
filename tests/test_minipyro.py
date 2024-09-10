import pytest
from pyroapi import pyro_backend

from effectful.handlers.minipyro import default_runner
from effectful.ops.handler import closed_handler


@pytest.fixture
def jit():
    return False


@pytest.fixture
def backend():
    with pyro_backend("effectful-minipyro"):
        with closed_handler(default_runner):
            yield


# noinspection PyUnresolvedReferences
from pyroapi.tests import *  # noqa: F401, E402, F403
