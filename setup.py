from setuptools import find_packages, setup


setup(
    name="signalgraph",
    version="0.1.0",
    description="Engine-first GraphRAG technical intelligence platform for research-to-production decisions.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["signalgraph=signalgraph.cli:main"]},
)
