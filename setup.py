from skbuild import setup

setup(
    name="rwkv_cpp",
    description="Python wrapper for rwkv.cpp",
    long_description_content_type="text/markdown",
    version="0.0.1",
    author="",
    author_email="",
    license="MIT",
    packages=["rwkv_cpp"],
    install_requires=[
        "numpy>=1.24.1",
        "torch>=2.0.0",
        "tokenizers>=0.13.3",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
