from setuptools import setup, find_packages

packages = find_packages('.')

version = "0.0.1"

setup(
    name="interp",
    version=version,
    license="MIT",
    packages=packages,
    include_package_data=True,
    zip_safe=True,
)
