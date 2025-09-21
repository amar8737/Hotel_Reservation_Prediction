from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()


setup(
    name="Hotel-Booking-Price-Prediction",
    version="1.0.0",
    author="Amar",
    packages=find_packages(),
    install_requires=requirements,
)
