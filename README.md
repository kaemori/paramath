# The Paramath Programming Language

Paramath is a Domain-Specific Language (DSL) that transforms procedural math code into mathematical expressions. Paramath compiles into math expressions with operations that are commonly found on standard scientific calculators, allowing evaluation of logical code on mathematical hardware.

All this, powered by Python!

> **Why the name "Paramath"? </br>**
> The word "paramath" comes from the portmanteau of "parenthesis" and "mathematics".

## Features

-   **S-expression syntax**: Clean, unambiguous structure
-   **Automatic optimization**: Duplicate detection and subexpression extraction
-   **Loop unrolling**: Compile-time iteration for performance
-   **Flexible output**: Display or store results in variables

## Documentation

Full documentation is available [here](docs/paramath_docs.md)!

## Installation

### Method 1 (PyPI):

To install the latest release, just install it fron PyPI using:

```
pip install paramath-lang
```

Afterwards, run this to ensure Paramath is correctly installed:

```
paramath --version
```

To update, just run:

```
pip install --upgrade paramath-lang
```

### Method 2 (Direct Git Clone):

Clone this repository by running the following commands in a shell:

```
git clone https://github.com/kaemori/paramath.git
cd paramath
```

Then, run the following command to install Paramath:

```
python -m pip install .
```

> Note: We recommend installing via PyPI for easier updates. Direct git installs are snapshots and require reinstalling for each update.

## Usage

The compiler can be accessed with the following command:

```
paramath [-h] [-v] [-o FILE] [-D] [-V] [-O] [-S] [-L FILE] [filepath]

positional arguments:
  filepath            Input paramath file

options:
  -h, --help          show this help message and exit
  -v, --version       prints the Paramath version number and exits
  -o, --output FILE   output file (default: math.txt)
  -D, --debug         enable debug output
  -V, --verbose       enable verbose output
  -O, --print-output  print out the compiled output
  -m, --math-output   format output for calculators (use ^, implicit multiplication, ANS)
  -S, --safe-eval     prints and blocks python code from evaluating and exits, used for safely running unknown scripts
  -L, --logfile FILE  write logs to FILE
```

This help message can also be displayed later by running `paramath --help`

## License

This project is licensed under the [MIT License](LICENSE).
