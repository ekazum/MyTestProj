# MyTestCondaProj

A sample Conda project demonstrating Python code using the math package to perform various mathematical computations.

## Description

This project showcases a simple Python application that uses the built-in `math` package to perform and display various mathematical computations including:
- Trigonometric functions (sin, cos, tan)
- Logarithmic and exponential functions
- Power and square root calculations
- Mathematical constants (π, e, τ)
- Rounding functions (ceil, floor)
- Factorial calculations
- Greatest Common Divisor (GCD)
- Hyperbolic functions (sinh, cosh, tanh)

## Setup

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ekazum/MyTestCondaProj.git
   cd MyTestCondaProj
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate mytestconda
   ```

## Usage

Run the main Python script:
```bash
python main.py
```

The script will output various mathematical computations to the console.

## Deactivating the Environment

When you're done, you can deactivate the Conda environment:
```bash
conda deactivate
```

## Removing the Environment

To remove the Conda environment completely:
```bash
conda env remove -n mytestconda
```
