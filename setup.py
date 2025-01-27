import sys

from setuptools import find_packages, setup

VERSION = "0.1.0"

try:
    long_description = open("README.rst", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

TORCH_REQUIRE = ["torch"]
PYRO_REQUIRE = TORCH_REQUIRE + ["pyro-ppl"]
DOCS_REQUIRE = [
    "setuptools",
    "sphinx",
    "sphinxcontrib-bibtex",
    "sphinx_rtd_theme",
    "myst-parser",
    "nbsphinx",
]
DEV_REQUIRE = (
    PYRO_REQUIRE
    + TORCH_REQUIRE
    + DOCS_REQUIRE
    + [
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "pytest-benchmark",
        "mypy",
        "black",
        "flake8",
        "isort",
        "nbval",
        "nbqa",
    ]
)

setup(
    name="effectful",
    version=VERSION,
    description="Metaprogramming infrastructure",
    long_description=long_description,
    packages=find_packages(include=["effectful", "effectful.*"]),
    author="Basis",
    url="https://www.basis.ai/",
    project_urls={
        #     "Documentation": "",
        "Source": "https://github.com/BasisResearch/effectful",
    },
    package_data={"effectful": ["py.typed"]},
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/source/requirements.txt`
        "typing_extensions",
        "dm-tree",
    ],
    extras_require={
        "torch": TORCH_REQUIRE,
        "pyro": PYRO_REQUIRE,
        "dev": DEV_REQUIRE,
        "docs": DOCS_REQUIRE,
    },
    python_requires=">=3.10",
    keywords="machine learning statistics probabilistic programming bayesian modeling pytorch",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # yapf
)
