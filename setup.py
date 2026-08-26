from setuptools import setup, find_packages

setup(
    name="spur-math",
    version="0.1.0",
    description="Noyaux mathématiques accélérés AVX2 — polynômes certifiés SPEAR",
    long_description=open("README.md",encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="bahira",
    packages=["spur_math"],
    package_data={"spur_math": ["*.dll","*.so"]},
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)
