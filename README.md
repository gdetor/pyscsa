# pySCSA

pySCSA is a Python implementation of the one-dimensional Semi-Classical Signal Analysis method proposed in [1]. The primary concept behind SCSA is to consider any pulse-shaped signal as Schrödinger's operator potential and utilize its discrete spectrum to analyze the signal. For more information regarding the mathematical principles and the algorithm, please refer to [1].

In this repository, you will find the following scripts:
  1. **scsa.py** A Python class that implements the basic one-dimensional SCSA algorithm
  2. **example_sech.py** An example of how to use the SCSA class on a sech function
  3. **example_gaussian.py** An example of how to use the SCSA class on a Gaussian function
  4. **example_neural_spike.py** An example of how to apply SCSA on a neural spike


## Example Usage

The user can run the demos/examples provided here by typing the following commands in a terminal:
```bash
$ python3 example_sech.py

$ python3 example_gaussian.py
```


## Dependencies - Requirements

The SCSA class and the example scripts require the following Python packages:
  - Numpy >= 1.26.4
  - Scipy >= 1.13.0
  - Matplotlib >= 3.5.1
  - Scikit-learn >= 1.4.1.post1


## Tested platforms
The software available in this repository has been tested on the following platforms:
  - Ubuntu 22.04.4 LTS
  - Python 3.10.12
  - GCC 11.4.0
  - x86_64


## References
  1. Laleg-Kirati, Taous-Meriem, Emmanuelle Crépeau, and Michel Sorine. "Semi-classical signal analysis." Mathematics of Control, signals, and Systems 25 (2013): 37-61.
