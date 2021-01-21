import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = cythonize([
    Extension(
        "skroute.heuristics.brute._base_brute_force._base_brute_force",
        sources=["skroute/heuristics/brute/_base_brute_force/_base_brute_force.pyx"]),
    Extension(
        "skroute.metaheuristics.genetics._base_genetics._utils_genetic",
        sources=["skroute/metaheuristics/genetics/_base_genetics/_utils_genetic.pyx"]),
    Extension(
        "skroute._utils._utils",
        sources=["skroute/_utils/_utils.pyx"]),
    Extension(
        "skroute.metaheuristics.genetics._base_genetics._base_genetic",
        sources=["skroute/metaheuristics/genetics/_base_genetics/_base_genetic.pyx"]),
    Extension(
        "skroute.metaheuristics.simulated_annealing._base_simulated_annealing._base_simulated_annealing",
        sources=["skroute/metaheuristics/simulated_annealing/_base_simulated_annealing/_base_simulated_annealing.pyx"]),
    Extension(
        "skroute.metaheuristics.simulated_annealing._base_simulated_annealing._utils_sa",
        sources=["skroute/metaheuristics/simulated_annealing/_base_simulated_annealing/_utils_sa.pyx"]),
    ])

HERE = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

with open(os.path.join(HERE, "requirements.txt")) as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="scikit-route", 
    version="1.0.0a2", 
    description="Compute Routes easy and fast",
    long_description=README, 
    long_description_content_type="text/markdown",  
    url="https://github.com/arubiales/scikit-route", 
    author="Alberto Rubiales", 
    author_email="al.rubiales.b@gmail.com", 
    license="MIT", 
    classifiers=[ 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    packages=find_packages(),
    setup_requires=[
        'setuptools>=18.0',
        "cython"
    ],
    install_requires=REQUIREMENTS,
    ext_modules = extensions,
    include_package_data=True,
    package_data={"": ["datasets/_data/_latitude_longitude/*.tsp", "datasets/*.txt", "datasets/_data/_money_cost/*.pkl"]},
)
