from setuptools import setup

setup(
    name="tdigest",
    version="1.0.0",
    description="Incremental calculation of quantiles",
    install_requires=[
        "numpy",
        "numba",
    ],
)
