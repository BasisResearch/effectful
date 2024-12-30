import importlib

import pytest
from pyroapi import pyro_backend

from effectful.ops.semantics import handler


@pytest.fixture
def jit():
    return False


@pytest.fixture
def backend():

    minipyro = importlib.import_module("docs.source.minipyro", package="effectful")

    with pyro_backend("effectful-minipyro"):
        with handler(minipyro.default_runner):
            yield


# noinspection PyUnresolvedReferences
from pyroapi.tests import *  # noqa: F401, E402, F403
