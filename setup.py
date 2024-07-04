import os
from setuptools import setup, find_packages

version = "v0.0.0"

setup(
    name="chattts",
    version=os.environ.get("CHTTS_VER", version).lstrip("v"),
    description="A generative speech model for daily dialogue",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="2noise",
    author_email="open-source@2noise.com",
    maintainer="fumiama",
    url="https://github.com/2noise/ChatTTS",
    packages=find_packages(include=["ChatTTS", "ChatTTS.*"]),
    package_data={
        "ChatTTS.res": ["homophones_map.json", "sha256_map.json"],
    },
    license="CC BY-NC 4.0",
    install_requires=[
        "numba",
        "numpy<2.0.0",
        "omegaconf>=2.3.0",
        "pybase16384",
        "torch>=2.1.0",
        "tqdm",
        "transformers>=4.41.1",
        "vector_quantize_pytorch",
        "vocos",
    ],
    platforms="any",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
