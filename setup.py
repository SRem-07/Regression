from setuptools import setup, find_packages # type: ignore

setup(
    name="linreg_pro",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
    ],
    python_requires=">=3.8",
)