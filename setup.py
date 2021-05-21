import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prism-style-transfer",
    version="0.1",
    author="Moritz Thuening",
    author_email="moritz.thuening@gmail.com",
    description=("High Resolution Style Transfer in PyTorch"
                 " with Color Control and Mixed Precision"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moritztng/prism",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tensorboard>=2.3.0',
        'Pillow>=7.0.0'
    ],
    entry_points={
        "console_scripts": [
            "style-transfer = style_transfer.__main__:main"
        ]
    }
)
