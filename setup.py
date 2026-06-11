from setuptools import find_packages, setup


setup(
    name="crest-retrieval",
    description="Core implementation of Crest compact retrieval with candidate-bounded residual repair",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
    ],
)
