from setuptools import setup
from candl import __author__, __version__

setup(
    name="candl_like",
    version=__version__,
    description="Differentiable Likelihood for CMB Analysis",
    url="https://github.com/Lbalkenhol/candl",
    author=__author__,
    license="MIT",
    packages=["candl"],
    package_data={
        "candl": [
            "transformations/*",
            "data/*",
            "data/*/**",
            "tests/*",
            "foreground_templates/*",
        ]
    },
    zip_safe=False,
)
