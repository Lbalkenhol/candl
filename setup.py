from setuptools import setup

setup(
    name="candl",
    version="0.0.1",
    description="candl",
    url="none",
    author="",
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
