from setuptools import setup, find_packages

setup(
    name="my-library",
    version="1.0.0",
    description="一个示例 Python 库",
    packages=find_packages(),
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
