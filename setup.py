import setuptools
# import versioneer

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hammock",
    version='0.0',
    author="Tiancheng Yang",
    author_email="t77yang@uwaterloo.ca",
    description="Hammock - visualization of categorical or mixed categorical/continuous data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TianchengY/hammock_plot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
    install_requires=["matplotlib", "numpy", "pandas"],
    # cmdclass=versioneer.get_cmdclass(),
)
