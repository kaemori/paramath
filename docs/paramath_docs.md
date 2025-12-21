# Paramath Language Specification v2.2.4

_A differentiable programming language for continuous computation_

---

## Table of Contents

1. [Introduction](#introduction)
2. [Syntax Overview](#syntax-overview)
3. [Data Types](#data-types)
4. [Operators and Functions](#operators-and-functions)
5. [Program Structure](#program-structure)
6. [Pragmas and Directives](#pragmas-and-directives)
7. [Function Definitions](#function-definitions)
8. [Lambda Expressions](#lambda-expressions)
9. [Loops and Iteration](#loops-and-iteration)
10. [Intermediates and Code Blocks](#intermediates-and-code-blocks)
11. [Advanced Features](#advanced-features)
12. [Complete Examples](#complete-examples)
13. [Command-Line Interface](#command-line-interface)
14. [Error Reference](#error-reference)
15. [Best Practices](#best-practices)

---

## Introduction

**Paramath** is a domain-specific language designed for applications that evaluate mathematical statements (more specifically, mathematical functions in the pre-calculus domain) instead of normal machine code.

> **Why the name "Paramath"?**  
> The word "paramath" comes from the portmanteau of "parenthesis" and "mathematics."

### Key Features

-   **S-expression syntax**: Clean, unambiguous structure
-   **Automatic optimization**: Duplicate detection and subexpression extraction
-   **Loop unrolling**: Compile-time iteration for performance
-   **Flexible output**: Display or store results in variables

---

## Syntax Overview

### Basic Structure

Paramath uses **S-expressions** (symbolic expressions) similar to Lisp:

```scheme
(operator operand1 operand2 ...)
```

### Comments

Comments start with `#` and continue to the end of the line:

```scheme
# This is a comment
(+ 2 3)  # This adds 2 and 3
```

### Variables

Reserved variable names (case-insensitive):

-   Single letters: `a`, `b`, `c`, `d`, `e`, `f`, `x`, `y`, `m`
-   Special: `ans` (previous result), `pi`, `e`

> **Note on `ans`:** The `ans` variable stores the result of the last computed expression. When using `//store X`, the value goes to both `X` and `ans`. However, when `//dupe true` is enabled (default), the compiler may create intermediate variables that make `ans` behave unpredictably—prefer using named intermediates instead!

---

## Data Types

### Numeric Literals

```scheme
42        # Integer
3.14159   # Float
-17       # Negative integer
2.5e-3    # Scientific notation (preferred for small numbers!)
```

### Constants

Define constants using `//global` anywhere in your program:

```scheme
//global MEANING 42
//global GOLDEN_RATIO (/ (+ 1 (** 5 0.5)) 2)
```

Constants can reference other previously defined constants:

```scheme
//global RADIUS 5
//global AREA (* pi (** RADIUS 2)
```

You can also set precision for individual constants:

```scheme
//precision 4
//global PI_APPROX pi  # Evaluates pi to 4 decimal places

//precision 10
//global E_PRECISE e   # Evaluates e to 10 decimal places
```

### Python Expression Evaluation

Both `//global` and `//local` support evaluating Python expressions! This lets you compute values using Python's standard library and previously defined globals/locals:

```scheme
# Simple arithmetic (evaluated as Python)
//global LENGTH len("hello world")

# Reference previously defined globals
//global RADIUS 5
//global AREA (globals.RADIUS ** 2 * 3.14159)

# In loops with //local, reference loop-scoped variables
//repeat 5 i
  //local squared locals.i ** 2
  //local doubled globals.LENGTH + i
  //ret (+ squared doubled)
//endrepeat
```

**Available Python Functions:**

-   All math functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `arcsin`, `arccos`, `arctan`, `abs`, `pi`, `e`
-   Built-in functions: `len()`, `str()`, `int()`, `float()`, etc.
-   Access to `globals` and `locals` as special namespaces

**Important Notes:**

-   Access globals with `globals.VARIABLE_NAME` (NOT `global.VARIABLE_NAME`)
-   Access locals with `locals.VARIABLE_NAME` (NOT `local.VARIABLE_NAME`)
-   Evaluate to numeric values that can be used in expressions

### Aliases

Create alternative names for variables:

```scheme
//alias input x        # "input" now refers to x
//alias output y       # "output" now refers to y
//alias theta a        # "theta" now refers to a

//display
//ret (sin theta)      # Uses variable a
```

---

## Operators and Functions

### Basic Arithmetic

| Operator | Description    | Example          |
| -------- | -------------- | ---------------- |
| `+`      | Addition       | `(+ 2 3)` → 5    |
| `-`      | Subtraction    | `(- 10 3)` → 7   |
| `*`      | Multiplication | `(* 4 5)` → 20   |
| `/`      | Division       | `(/ 10 2)` → 5   |
| `**`     | Exponentiation | `(** 2 8)` → 256 |

> **Note:** `+` and `*` support multiple arguments and are right-associative:
>
> ```scheme
> (+ 1 2 3 4)  # Expands to (+ 1 (+ 2 (+ 3 4)))
> ```

### Trigonometric Functions

| Function | Description     |
| -------- | --------------- |
| `sin`    | Sine            |
| `cos`    | Cosine          |
| `tan`    | Tangent         |
| `arcsin` | Inverse sine    |
| `arccos` | Inverse cosine  |
| `arctan` | Inverse tangent |
| `abs`    | Absolute value  |

Example:

```scheme
(sin (* pi (/ x 2)))  # sin(πx/2)
```

### Logical Operations

All logical operations are continuous approximations that depend on epsilon:

| Operation     | Syntax     | Description                       |
| ------------- | ---------- | --------------------------------- |
| Equality      | `(== a b)` | Returns ~1 when a≈b, ~0 otherwise |
| Equal to zero | `(=0 a)`   | Returns ~1 when a≈0, ~0 otherwise |
| Greater than  | `(> a b)`  | Returns ~1 when a>b, ~0 otherwise |
| Greater/equal | `(>= a b)` | Returns ~1 when a≥b, ~0 otherwise |
| Less than     | `(< a b)`  | Returns ~1 when a<b, ~0 otherwise |
| Less/equal    | `(<= a b)` | Returns ~1 when a≤b, ~0 otherwise |
| Logical not   | `(! a)`    | Returns 1-a                       |

> **Tip:** The smaller your epsilon, the sharper these comparisons become. Recommended: `1e-99` for maximum sharpness.

### Mathematical Operations

| Operation | Syntax      | Description                      |
| --------- | ----------- | -------------------------------- |
| Sign      | `(sign a)`  | Returns -1, 0, or 1 (continuous) |
| Max       | `(max a b)` | Maximum of a and b               |
| Min       | `(min a b)` | Minimum of a and b               |
| Max0      | `(max0 a)`  | Maximum of a and 0               |
| Min0      | `(min0 a)`  | Minimum of a and 0               |
| Modulo    | `(mod a b)` | a modulo b                       |
| Fraction  | `(frac a)`  | Fractional part of a             |
| Natural   | `(nat a)`   | Rounds a to nearest integer      |

### Conditional Operations

```scheme
(if condition then_value else_value)
```

Example:

```scheme
(if (== 0 (mod x 2))
    (/ x 2)
    (+ (* 3 x) 1))  # Collatz conjecture!
```

---

## Program Structure

A Paramath program consists of:

1. **Aliases** (optional): Alternative variable names
2. **Configuration** (optional): Precision, epsilon, optimization settings
3. **Globals** (optional): Named constant definitions
4. **Functions** (optional): User-defined functions
5. **Code blocks**: Computation blocks with output directives

### Minimal Program

```scheme
//display
//ret (+ 2 2)
```

> **Note:** The final surrounding parentheses are not necessarily needed if you're using a single-line string, but if any error occurs, surrounding the expression with parentheses usually resolves it.)

### Complete Program Structure

```scheme
# Aliases
//alias r x

# Configuration
//precision 6
//epsilon 1e-99
//simplify true
//dupe true 0
//sympy false

# Constants
//global RADIUS 5

# Functions
//def circle_area $r
//ret (* pi (* $r $r))

# Computation
//display
//ret (circle_area RADIUS)
```

---

## Pragmas and Directives

Pragmas are special commands that start with `//`.

### Configuration Pragmas

Can appear anywhere and affect subsequent code:

#### `//precision`

Sets decimal precision for constant evaluation:

```scheme
//precision 4
//global PI_LOW pi  # Evaluates to 4 decimals

//precision 10
//global PI_HIGH pi  # Evaluates to 10 decimals
```

> **Note:** `pi` is calculated using python's math module, which provides up to 15 accurate significant decimals, and reaches 17 decimal places. If a higher precision is needed, creating a custom `//global PI_ACCURATE` would be the most elegant solution. Note that using `pi` as a variable instead is also recommended, as the compiler assumes the `pi` variable instead of the literal on the calculator, resulting in less characters in the final output (e.g. ` (* 4 pi)` instead of `(* 4 3.1415926535)`)

#### `//epsilon`

Sets the epsilon parameter for logical operations:

```scheme
//epsilon 1e-99  # Sharp comparisons
//epsilon 1e-50  # Slightly softer
```

> **Note:** Smaller epsilon = sharper decision boundaries. Recommended: `1e-99` for most use cases.

#### `//variables`

Sets the letter variables that the compiler recognizes and accepts. Note that the variables `ans`, `pi`, and `e` cannot be changed.

```scheme
//variables a b c d e f x y z m  # Default
//variables m n o p q
```

#### `//simplify`

Enables/disables compile-time simplification of literal expressions:

```scheme
//simplify true   # Default: simplifies (+ 2 3) to 5 at compile time
//simplify false  # Keeps all expressions as-is
```

#### `//dupe`

Enables/disables duplicate subexpression detection, with optional second parameter for minimum savings threshold:

```scheme
//dupe true       # Default: extracts repeated subexpressions
//dupe false      # Keeps expression as-is, even with duplicates
//dupe true -999  # Makes sure all possible duplications are extracted
```

> **Warning:** When `//dupe true`, the compiler may create intermediate results automatically. This makes `ans` less predictable - use named variables instead!

#### `//sympy`

Enables/disables using Sympy to simplify the math expression:

```scheme
//sympy true  # Will use SymPy to simplify expressions
//sympy false  # Default. Disables using SymPy to simplify expressions
```

> **Warning:** This option is not extensively tested, and thus is more or less experimental. To use it, you must first have SymPy installed, otherwise the compiler will crash.

### Output Directives

Every code block must have an output directive:

#### `//display`

Output result to console:

```scheme
//display
//ret (+ 2 3)
```

#### `//store`

Store result in a variable:

```scheme
//store x
//ret (+ 2 3)

//display
//ret (* x 2)  # Uses stored value
```

Default variable names: `a`, `b`, `c`, `d`, `e`, `f`, `x`, `y`, `m`

---

## Function Definitions

### Syntax

```scheme
//def function_name $param1 $param2 ...
//ret (function body)
```

-   Function names are case-insensitive
-   Parameters must start with `$`
-   Body must end with with `//ret` (return directive)

### Examples

```scheme
//def square $x
//ret (* $x $x)

//def distance $x1 $y1 $x2 $y2
//ret (** (+ (** (- $x2 $x1) 2) (** (- $y2 $y1) 2)) 0.5)

//display
//ret (distance 0 0 3 4)  # Returns 5.0
```

> **Note:** Recursion is strictly prohibited; This is because functions are unwrapped at compile time, and it will unwrap until Python raises a `RecursionError` error.

---

## Lambda Expressions

Anonymous functions for one-time use.

### Syntax

```scheme
((lambda ($param1 $param2 ...) body) arg1 arg2 ...)
```

### Examples

```scheme
# Simple lambda
((lambda ($x) (* $x $x)) 5)  # Returns 25

# Multiple parameters
((lambda ($x $y) (+ $x $y)) 3 4)  # Returns 7

# Nested lambdas
((lambda ($f $x) ($f ($f $x)))
 (lambda ($y) (* $y 2))
 5)  # Returns 20 (doubles 5 twice)
```

### Lambda Rules

-   Must be immediately applied (no storing lambdas)
-   Parameters must start with `$`
-   Body is a single expression (no `//ret` needed!)

> **Note:** I mainly added this feature as a joke and a tribute to lisp. This has no practical use cases whatever, but feel free to use it!

### Lambda Assignments with Walrus Operator

You can assign lambdas to variables for reuse **within the same code block** using the `:=` (walrus) operator:

```scheme
//display
double := (lambda ($x) (* $x 2))
triple := (lambda ($x) (* $x 3))

//ret (+ (double 5) (triple 5))  # Returns 25 (10 + 15)
```

**How It Works:**

-   Lambdas are stored as **lambda intermediates** using `:=` instead of `=`
-   They can be called like regular functions in the same code block
-   Each lambda call gets expanded and substituted at compile time

**Example with Nested Lambdas:**

```scheme
//display
compose := (lambda ($f $g $x) ($f ($g $x)))
add_one := (lambda ($n) (+ $n 1))
double := (lambda ($n) (* $n 2))

//ret (compose double add_one 5)  # ((double (add_one 5))) = 12
```

---

## Loops and Iteration

### Basic Loop Syntax

```scheme
//repeat count_expression
  # Loop body
//endrepeat
```

> **Note:** Loops are **unrolled at compile time** - the loop body is copied `count` times with substitutions.

### Simple Loop

```scheme
//global SIZE 3

//display
//repeat globals.SIZE
   //display
   //ret (+ x 1)
//endrepeat
```

This generates 3 separate expressions, each computing `(+ x 1)`.

> **Note:** Indentations are not necessary, but they help with readability.

### Loops with Iterator Variable

```scheme
//repeat 5 i
  //display
  //ret (* i i)
//endrepeat
```

The iterator `i` takes values 0, 1, 2, 3, 4 in each unrolled iteration.

### Local Variables in Loops

Use `//local` to define variables that only exist within the loop:

```scheme
//display
//repeat 3 i
  //local square (* i i)

  //display
  //ret (+ square 1)
//endrepeat
```

Each iteration gets its own `square` variable with the correct value.

### Nested Loops

Loops can be nested:

```scheme
//display
//repeat 3 i
  //repeat 3 j
    //ret (+ (* i 10) j)
  //endrepeat
//endrepeat
```

### Advanced Loop Example

```scheme
//global N 10

//display
//repeat globals.N i
  //local x_val (+ i 1)
  //local y_val (* x_val x_val)
  //ret (/ y_val x_val)
//endrepeat
```

---

## Intermediates and Code Blocks

### Intermediate Assignments

Instead of nested parentheses hell, you can break expressions into named parts:

```scheme
//display
a = (** x 2)
b = (** y 2)
c = (+ a b)
//ret (** c 0.5)
```

This is **much** cleaner than `(** (+ (** x 2) (** y 2)) 0.5)`!

### Naming Rules

-   Intermediate names must be alphanumeric (plus underscores)
-   They cannot conflict with:
    -   Global constants
    -   Function names
    -   Aliases
    -   Variable names
    -   Other intermediates in the same scope

### Complex Example

```scheme
//display
# Compute quadratic formula
discriminant = (- (** b 2) (* 4 (* a c)))
sqrt_disc = (** discriminant 0.5)
numerator = (+ (* -1 b) sqrt_disc)
denominator = (* 2 a)
//ret (/ numerator denominator)
```

### Automatic Expansion

The compiler automatically expands intermediates into the final expression. You write clean code, it generates optimized math!

---

## Advanced Features

### Automatic Duplicate Detection

When `//dupe true` (default), the compiler finds repeated subexpressions and extracts them automatically:

```scheme
//dupe true -999
//display
//ret (+ (* (+ x 1) (+ x 1)) (* (+ x 1) (+ x 1)))
```

The compiler recognizes `(+ x 1)` appears 4 times and may create an intermediate for it.

### Epsilon in Expressions

Access the current epsilon value using `ε` or `epsilon`:

```scheme
//epsilon 0.01
//display
//ret (/ x (+ x ε))  # Uses current epsilon value
```

### Expression Length Optimization

The compiler tracks expression length and tries to minimize it:

-   Extracts beneficial duplicates (saves space)
-   Flattens associative operations
-   Applies heuristics to simplify (e.g., `x + x + x` → `x * 3`)
-   Applies identity simplifications (e.g., `x * 1` → `x`)

---

## Complete Examples

### Example 1: Polynomial Evaluation with Horner's Method

```scheme
//epsilon 1e-99

# Evaluate polynomial: ax³ + bx² + cx + d
//def horner $x $a $b $c $d
  term1 = (* $a $x)
  term2 = (+ term1 $b)
  term3 = (* term2 $x)
  term4 = (+ term3 $c)
  term5 = (* term4 $x)
  //ret (+ term5 $d)

//display
//ret (horner x 2 -3 1 5)
```

### Example 2: Fibonacci with Loops

```scheme
//global N 10

//repeat globals.N i
  //display
  special = < i 2
  //ret (if special i (** ans 2))
//endrepeat
```

### Example 3: Vector Operations

```scheme
//alias x1 a
//alias y1 b
//alias x2 c
//alias y2 d

//def dot_product $ax $ay $bx $by
//ret (+ (* $ax $bx) (* $ay $by))

//def magnitude $x $y
//ret (** (+ (* $x $x) (* $y $y)) 0.5)

//display
vec1_mag = (magnitude x1 y1)
vec2_mag = (magnitude x2 y2)
dot = (dot_product x1 y1 x2 y2)
//ret (/ dot (* vec1_mag vec2_mag))  # Cosine of angle
```

### Example 4: Smooth Activation Functions

```scheme
//epsilon 1e-99

//def sigmoid $x
//ret (/ 1 (+ 1 (** e (* -1 $x))))

//def tanh_approx $x
//ret (- (* 2 (sigmoid (* 2 $x))) 1)

//def relu $x
//ret (max0 $x)

//def gelu $x
  sig = (sigmoid (* 1.702 $x))
  //ret (* $x sig)

//display
//ret (gelu x)
```

### Example 5: Numerical Differentiation

```scheme
//global H 1e-5

//def derivative $f $x
  x_plus_h = (+ $x H)
  x_minus_h = (- $x H)
  f_plus = ($f x_plus_h)
  f_minus = ($f x_minus_h)
  //ret (/ (- f_plus f_minus) (* 2 H))

//def my_function $x
//ret (* $x (* $x $x))  # x³

//display
//ret (derivative my_function x)  # Should be ~3x²
```

### Example 6: Parametric Curves

```scheme
//global N 50

//repeat globals.N i
  //local t (/ i N)
  //local x_pos (cos (* 2 (* pi t)))
  //local y_pos (sin (* 2 (* pi t)))
  //display
  //ret (+ (* x_pos x_pos) (* y_pos y_pos))  # Should be 1 (circle!)
//endrepeat
```

---

## Command-Line Interface

### Basic Usage

```bash
paramath [options] filepath
```

### Options

| Flag      | Long Form        | Description                                           | Example                            |
| --------- | ---------------- | ----------------------------------------------------- | ---------------------------------- |
| `-h`      | `--help`         | Show help message                                     | `paramath --help`                  |
| `-v`      | `--version`      | Print version and exit                                | `paramath --version`               |
| `-o FILE` | `--output FILE`  | Output file (default: `math.txt`)                     | `paramath input.pm -o results.txt` |
| `-D`      | `--debug`        | Enable debug output                                   | `paramath input.pm -D`             |
| `-V`      | `--verbose`      | Enable verbose output                                 | `paramath input.pm -V`             |
| `-O`      | `--print-output` | Print compiled output to console                      | `paramath input.pm -O`             |
| `-S`      | `--safe-eval`    | Block Python code evaluation and print what would run | `paramath unknown.pm -S`           |
| `-L FILE` | `--logfile FILE` | Write logs to FILE                                    | `paramath input.pm -L debug.log`   |

### Flag Details

#### `-O` / `--print-output`

Prints the compiled mathematical expressions to stdout:

```bash
$ paramath myfile.pm -O
reading myfile.pm

to ('display', None):
(sin(x))+1
to ('display', None):
(cos(x))+2

=== compilation successful! ===
generated 2 expressions
written to: math.txt
```

#### `-S` / `--safe-eval`

**Security feature**: Blocks execution of Python code in `//global` and `//local` expressions. This lets you safely inspect unknown paramath files before running them:

```bash
$ paramath untrusted.pm -S
reading untrusted.pm
[safe evaluation enabled]
# ... would print what Python code would execute, then exit without running it
```

Use this when:

-   Running untrusted `.pm` files
-   Reviewing user-submitted paramath code
-   Auditing what Python expressions will be evaluated

#### `-D` / `--debug`

Enables detailed debug output showing parsing steps, optimization decisions, etc. Useful for understanding how the compiler works.

#### `-V` / `--verbose`

Enables verbose output showing high-level compilation progress.

#### `-L` / `--logfile FILE`

Writes all debug and verbose output to a file instead of stdout:

```bash
paramath myfile.pm -D -V -L compiler.log
```

### Examples

```bash
# Compile with verbose output and save to custom file
paramath myprogram.pm -Vo results.txt

# Compile, print output, and save debug log
paramath myprogram.pm -ODL debug.log

# Check unknown file before running (safe eval)
paramath untrusted.pm -S

# Minimal compilation (quiet)
paramath myprogram.pm
```

---

## Error Reference

### Parser Errors

| Error                                     | Cause                                  | Solution                                 |
| ----------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `unexpected end of tokens`                | Missing closing parenthesis            | Add `)`                                  |
| `unexpected closing parenthesis`          | Extra `)` or missing `(`               | Check balance                            |
| `unknown identifier 'X'`                  | Undefined variable/constant            | Define with `//global` or check spelling |
| `unknown operation 'X'`                   | Undefined function                     | Define with `//def` or check spelling    |
| `incomplete expression at line X`         | Unbalanced parentheses in intermediate | Check your assignment                    |
| `expression without assignment at line X` | Forgot `=` in intermediate             | Use `var_name = expr` format             |

### Block Errors

| Error                                | Cause                                | Solution             |
| ------------------------------------ | ------------------------------------ | -------------------- |
| `no output mode before codeblock`    | Missing `//display` or `//store`     | Add output directive |
| `codeblock has no //ret directive`   | Missing `//ret`                      | Add `//ret expr`     |
| `incomplete //ret expression`        | Unbalanced parentheses after `//ret` | Check parentheses    |
| `'X' conflicts with global constant` | Naming conflict                      | Use different name   |
| `'X' conflicts with function name`   | Naming conflict                      | Use different name   |

### Loop Errors

| Error                                            | Cause                  | Solution                             |
| ------------------------------------------------ | ---------------------- | ------------------------------------ |
| `//repeat without matching //endrepeat`          | Missing `//endrepeat`  | Add `//endrepeat`                    |
| `//endrepeat without matching //repeat`          | Extra `//endrepeat`    | Remove it                            |
| `repeat range must be non-negative`              | Negative loop count    | Use positive number                  |
| `//local can only be used inside //repeat loops` | `//local` outside loop | Move inside loop or use intermediate |
| `'X' already defined in this loop`               | Duplicate local name   | Use different name                   |

### Function Errors

| Error                                     | Cause                  | Solution                  |
| ----------------------------------------- | ---------------------- | ------------------------- |
| `function 'X' expects N arguments, got M` | Wrong argument count   | Check function definition |
| `function parameters must start with $`   | Invalid parameter name | Use `$param` format       |

### Lambda Errors

| Error                                 | Cause                | Solution                    |
| ------------------------------------- | -------------------- | --------------------------- |
| `lambda must be immediately applied`  | Standalone lambda    | Wrap: `((lambda ...) args)` |
| `lambda expects N arguments, got M`   | Wrong argument count | Check parameters            |
| `lambda parameters must start with $` | Invalid parameter    | Use `$param` format         |

---

## Best Practices

### 1. Use Small Epsilon Values

```scheme
//epsilon 1e-99  # Recommended for sharp comparisons
```

> **Tip:** Avoid epsilon values larger than `1e-50` unless you specifically need smooth transitions.

### 2. Prefer Intermediates Over Deeply Nested Expressions

```scheme
# Bad
//ret (/ (+ (** (- x2 x1) 2) (** (- y2 y1) 2)) (+ (abs (- x2 x1)) (abs (- y2 y1))))

# Good
dx = (- x2 x1)
dy = (- y2 y1)
euclidean = (** (+ (* dx dx) (* dy dy)) 0.5)
manhattan = (+ (abs dx) (abs dy))
//ret (/ euclidean manhattan)
```

### 3. Use Aliases for Semantic Clarity

```scheme
//alias input_voltage x
//alias output_current y
//alias resistance a

//display
//ret (/ input_voltage resistance)  # Ohm's law
```

### 4. Avoid Relying on `ans` with Dupe Detection

When `//dupe true`, the compiler may create hidden intermediates that clobber `ans`. Instead:

```scheme
# Questionable
//store ans
//ret (complex_computation x)

//display
//ret (* ans 2)  # Might not be what you expect!

# Better
//alias result1 a

//store result1
//ret (complex_computation x)

//display
//ret (* result1 2)  # Explicit and clear
```

### 5. Use Loops for Repeated Patterns

```scheme
# Bad: copy-paste hell
//display
//ret (* 0 0)
//display
//ret (* 1 1)
//display
//ret (* 2 2)
# ... 97 more times

# Good: loop
//display
//repeat 100 i
  //ret (** i 2)
//endrepeat
```

### 6. Break Complex Functions into Smaller Ones

```scheme
# Bad: monolithic function
//def neural_network $x
//ret (... 50 lines of nested operations ...)

# Good: modular design
//def sigmoid $x
//ret (/ 1 (+ 1 (** e (* -1 $x))))

//def layer $x $w $b
//ret (sigmoid (+ (* $w $x) $b))

//def neural_network $x
  h1 = (layer $x 0.5 0.1)
  h2 = (layer h1 0.3 -0.2)
  //ret (layer h2 0.8 0.0)
```

### 7. Comment Your Epsilon Choices

```scheme
# Sharp comparison for exact equality check
//epsilon 1e-99
//display
//ret (== x y)

# Smooth transition for gradient-based optimization
//epsilon 1e-10
//display
//ret (if (> x threshold) high_val low_val)
```

### 8. Use Meaningful Constant Names

```scheme
# Bad
//global C1 299792458
//global C2 6.62607015e-34

# Good
//global SPEED_OF_LIGHT 299792458
//global PLANCK_CONSTANT 6.62607015e-34
```

---

## Language Grammar (EBNF)

```ebnf
program        ::= (pragma | function | code_block)*

pragma         ::= "//" pragma_name pragma_args?
pragma_name    ::= "precision" | "epsilon" | "global" | "alias" |
                   "display" | "store" | "ret" | "simplify" | "dupe" | "variables" |
                   "def" | "enddef" | "repeat" | "endrepeat" | "local"

code_block     ::= output_directive intermediate* ret_directive
output_directive ::= "//display" | "//store" var_name
intermediate   ::= identifier "=" expression newline
ret_directive  ::= "//ret" expression newline?

function       ::= "//def" func_name param* newline
                   intermediate* ret_directive newline
                   "

loop           ::= "//repeat" expression identifier? newline
                   (pragma | local_def | intermediate | ret_directive)*
                   "//endrepeat"

local_def      ::= "//local" identifier expression newline

expression     ::= number | variable | constant | "ε" | "epsilon" |
                   "(" operator expression* ")" |
                   lambda_expr

lambda_expr    ::= "(" "(" "lambda" "(" param* ")" expression ")"
                   expression* ")"

operator       ::= "+" | "-" | "*" | "/" | "**" |
                   "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan" |
                   "abs" | "==" | "=0" | ">" | ">=" | "<" | "<=" |
                   "!" | "sign" | "max" | "min" | "max0" | "min0" |
                   "if" | "mod" | "frac" | "nat" | func_name

param          ::= "$" identifier
var_name       ::= "a".."f" | "x" | "y" | "m"
number         ::= "-"? digit+ ("." digit+)? (("e"|"E") ("-"|"+")? digit+)?
identifier     ::= (alpha | "_") (alpha | digit | "_")*
```

---

## Contributing

Paramath is an evolving language. Contributions and feedback are welcome!

**Version**: 2.2.4
**License**: MIT  
**GitHub**: https://github.com/kaemori/paramath

---

_Last updated: December 2025_
