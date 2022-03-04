from setuptools import setup, find_packages

setup(
    name="tinyeos",
    version="2.0",
    description="equations of state and opacities for planets",
    url="",
    author="Simon Müller",
    author_email="simon.mueller7@uzh.ch",
    license="MIT",
    packages=find_packages(include=["tinyeos", "tinyeos.*"]),
    package_data={
        "tinyeos": [
            "data/eos/tables/*",
            "data/eos/interpolants/*",
            "data/kap/tables/*",
            "data/kap/interpolants/*",
        ]
    },
    install_requires=["fortranformat", "numpy", "scipy", "scikit-learn"],
)
