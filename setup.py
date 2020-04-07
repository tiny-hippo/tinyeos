from setuptools import setup, find_packages

setup(
    name="tinyeos",
    version="1.0",
    description="equations of state for planets",
    url="",
    author="Simon Müller",
    author_email="simon.mueller7@uzh.ch",
    license="MIT",
    packages=find_packages(include=["tinyeos", "tinyeos.*"]),
    package_data={"tinyeos": ["data/tables/*", "data/interpolants/*"]},
    install_requires=["fortranformat", "numpy", "numba", "scipy", "scikit-learn"]
)
