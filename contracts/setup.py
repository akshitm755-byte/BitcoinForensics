from setuptools import setup, find_packages

setup(
    name="abi_contracts",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["pydantic>=2.0.0"],
)
