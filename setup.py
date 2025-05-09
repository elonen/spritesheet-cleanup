from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spritesheet-cleanup",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "spritesheet-cleanup=pixelart.main:main",
        ],
    },
    author="Jarno Elonen",
    author_email="elonen@iki.fi",
    description="Pixel Art Restoration Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    license="MIT",
    keywords="pixel-art, sprite, restoration",
    url="https://github.com/elonen/spritesheet-cleanup",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.11",    
)
