from setuptools import setup, find_packages

setup(
    name="learn2slither",
    description="A AI agent trained to play the game snake. "
    "Train your own snake or test a pre-trained one.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    version="1.0.0",
    install_requires=["numpy", "torch", "pygame"],
    author="Joel Burleson",
    url="https://github.com/fburleson/learn2slither",
    python_requires=">=3.10",
)
