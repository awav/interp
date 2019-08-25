from setuptools import setup, find_packages

packages = find_packages('.')

setup(
    name="interp",
    license="MIT",
    packages=packages,
    include_package_data=True,
    zip_safe=True,
)
