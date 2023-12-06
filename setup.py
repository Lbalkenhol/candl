from setuptools import setup
from candl import __author__, __version__

setup(
    name="candl",
    version=__version__,
    description="candl",
    url="none",
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
