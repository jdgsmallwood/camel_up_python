from setuptools import setup

setup(
    name="camel-up",
    version="1.0",
    install_requires=[
        "black",
        "isort",
        "pyright",
        "jupyterlab",
        "pytest",
        "pytest-cov",
        "loguru",
        "pycln",
    ],
    extras_require={},
)
