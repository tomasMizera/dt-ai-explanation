import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TME", # Replace with your own username
    version="0.0.1",
    author="Tomas Mizera",
    author_email="tomas.mizera2@gmail.com",
    description="Explanator for machine learning text models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomasMizera/dt-ai-explanation",
    project_urls={
        "Bug Tracker": "https://github.com/tomasMizera/dt-ai-explanation/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "tme"},
    packages=setuptools.find_packages(where="tme"),
    python_requires=">=3.9",
)