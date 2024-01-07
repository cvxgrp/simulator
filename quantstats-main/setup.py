#!/usr/bin/env python

"""QuantStats: Portfolio analytics for quants
https://github.com/ranaroussi/quantstats
QuantStats performs portfolio profiling, to allow quants and
portfolio managers to understand their performance better,
by providing them with in-depth analytics and risk metrics.
"""

# from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.rstrip() for line in f]

setup(
    name="QuantStats",
    description="Portfolio analytics for quants",
    url="https://github.com/ranaroussi/quantstats",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    license="Apache Software License",
    platforms=["any"],
    keywords="""quant algotrading algorithmic-trading quantitative-trading
                quantitative-analysis algo-trading visualization plotting""",
    # packages=find_packages(exclude=["docs", "tests"]),
    packages=find_packages(include=["quantstats", "quantstats.*"]),
    install_requires=requirements,
    include_package_data=True,
    package_data={"static": ["quantstats/report.html"]},
)
