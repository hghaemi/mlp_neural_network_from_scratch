from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlp_neural_network_from_scratch",
    version="1.0.0",
    author="M. Hossein Ghaemi",
    author_email="h.ghaemi.2003@gmail.com",
    description="A comprehensive Multi-Layer Perceptron implementation with backpropagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hghaemi/mlp_neural_network_from_scratch.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlp-demo=examples.basic_example:main",
        ],
    },
    keywords="neural-network, machine-learning, deep-learning, mlp, backpropagation",       
)