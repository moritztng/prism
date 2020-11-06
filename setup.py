import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prism-style-transfer",
    version="0.0.1",
    author="Moritz Thuening",
    author_email="hello@prism.art",
    description="Simple Style Transfer",
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
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'tensorboard==2.3.0',
        'Pillow==7.2.0'
    ]
)
