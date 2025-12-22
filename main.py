"""
Sample Python script demonstrating math package usage.
Performs various mathematical computations and prints results to console.
"""

import math


def main():
    """Execute various mathematical computations using the math package."""
    print("=" * 50)
    print("Mathematical Computations Demo")
    print("=" * 50)
    
    # Basic trigonometric functions
    angle = 45  # degrees
    angle_rad = math.radians(angle)
    print(f"\nTrigonometric functions for {angle} degrees:")
    print(f"  sin({angle}°) = {math.sin(angle_rad):.4f}")
    print(f"  cos({angle}°) = {math.cos(angle_rad):.4f}")
    print(f"  tan({angle}°) = {math.tan(angle_rad):.4f}")
    
    # Logarithmic and exponential functions
    number = 10
    print(f"\nLogarithmic and exponential functions:")
    print(f"  log({number}) = {math.log(number):.4f}")
    print(f"  log10({number}) = {math.log10(number):.4f}")
    print(f"  exp(2) = {math.exp(2):.4f}")
    
    # Power and square root
    base = 4
    exponent = 3
    print(f"\nPower and roots:")
    print(f"  {base}^{exponent} = {math.pow(base, exponent):.4f}")
    print(f"  sqrt(16) = {math.sqrt(16):.4f}")
    print(f"  cbrt(27) = {math.pow(27, 1/3):.4f}")
    
    # Constants
    print(f"\nMathematical constants:")
    print(f"  π (pi) = {math.pi:.6f}")
    print(f"  e = {math.e:.6f}")
    print(f"  τ (tau) = {math.tau:.6f}")
    
    # Ceiling and floor functions
    value = 7.3
    print(f"\nRounding functions for {value}:")
    print(f"  ceil({value}) = {math.ceil(value)}")
    print(f"  floor({value}) = {math.floor(value)}")
    
    # Factorial
    n = 5
    print(f"\nFactorial:")
    print(f"  {n}! = {math.factorial(n)}")
    
    # GCD (Greatest Common Divisor)
    a, b = 48, 18
    print(f"\nGreatest Common Divisor:")
    print(f"  gcd({a}, {b}) = {math.gcd(a, b)}")
    
    # Hyperbolic functions
    x = 1
    print(f"\nHyperbolic functions for x={x}:")
    print(f"  sinh({x}) = {math.sinh(x):.4f}")
    print(f"  cosh({x}) = {math.cosh(x):.4f}")
    print(f"  tanh({x}) = {math.tanh(x):.4f}")
    
    print("\n" + "=" * 50)
    print("Computations completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
