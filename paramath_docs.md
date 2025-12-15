# Paramath Language Specification v2.0

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
13. [Error Reference](#error-reference)
14. [Best Practices](#best-practices)

---

## Introduction

**Paramath** is a domain-specific language designed for applications that evaluate mathematical statements (more specifically, mathematical functions in the pre-calculus domain) instead of normal machine code. To achieve this, Paramath uses smooth and continuous approximations of logical operations, making all computations differentiable.

> **Why the name "Paramath"?**  
> The word "paramath" comes from the portmanteau of "parenthesis" and "mathematics."

### Key Features

-   **Differentiable logic**: All operations remain continuous
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
//global AREA (* pi (* RADIUS RADIUS))
```

You can also set precision for individual constants:

```scheme
//precision 4
//global PI_APPROX pi  # Evaluates pi to 4 decimal places

//precision 10
//global E_PRECISE e   # Evaluates e to 10 decimal places
```

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

### Logical Operations (Differentiable)

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
| Modulo    | `(mod a b)` | a modulo b (differentiable!)     |
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

1. **Configuration** (optional): Precision, epsilon, optimization settings
2. **Globals** (optional): Named constant definitions
3. **Aliases** (optional): Alternative variable names
4. **Functions** (optional): User-defined functions
5. **Code blocks**: Computation blocks with output directives

> **Note:** Globals were named Constants before v0.4. Unlike Constants in v0.4, configurations and globals can appear anywhere! They only affect code that comes after them.

### Minimal Program

```scheme
//display
//ret (+ 2 2)
```

> **Note:** The final surrounding parentheses are not necessarily needed if you're using a single-line string, but if any error occurs, surrounding the expression with parentheses usually resolves it.

### Complete Program Structure

```scheme
# Configuration
//precision 6
//epsilon 1e-99
//simplify true
//dupe true

# Constants
//global RADIUS 5

# Aliases
//alias r x

# Functions
//def circle_area $r
//ret (* pi (* $r $r))
//enddef

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

> **Note:** `pi` is calculated using python's math module, which provides up to 15 accurate significant decimals, and reaches 17 decimal places. If a higher precision is needed, creating a custom `//global PI_ACCURATE` would be the most elegant solution.

#### `//epsilon`

Sets the epsilon parameter for logical operations:

```scheme
//epsilon 1e-99  # Sharp comparisons
//epsilon 1e-50  # Slightly softer
```

> **Note:** Smaller epsilon = sharper decision boundaries. Recommended: `1e-99` for most use cases.

#### `//simplify`

Enables/disables compile-time simplification of literal expressions:

```scheme
//simplify true   # Default: simplifies (+ 2 3) to 5 at compile time
//simplify false  # Keeps all expressions as-is
```

#### `//dupe`

Enables/disables duplicate subexpression detection:

```scheme
//dupe true   # Default: extracts repeated subexpressions
//dupe false  # Keeps expression as-is, even with duplicates
```

> **Warning:** When `//dupe true`, the compiler may create intermediate results automatically. This makes `ans` less predictable - use named variables instead!

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

Valid variable names: `a`, `b`, `c`, `d`, `e`, `f`, `x`, `y`, `m`

---

## Function Definitions

### Syntax

```scheme
//def function_name $param1 $param2 ...
//ret (function body)
//enddef
```

-   Function names are case-insensitive
-   Parameters must start with `$`
-   Body starts with `//ret` (return directive)

### Examples

```scheme
//def square $x
//ret (* $x $x)
//enddef

//def distance $x1 $y1 $x2 $y2
//ret (** (+ (** (- $x2 $x1) 2) (** (- $y2 $y1) 2)) 0.5)
//enddef

//display
//ret (distance 0 0 3 4)  # Returns 5.0
```

### Recursion

Functions can call themselves:

```scheme
//def factorial $n
//ret (if (<= $n 1)
        1
        (* $n (factorial (- $n 1))))
//enddef
```

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
//repeat SIZE
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
//repeat N i
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
a = (* x x)
b = (* y y)
c = (+ a b)
//ret (** c 0.5)
```

This is **much** cleaner than `(** (+ (* x x) (* y y)) 0.5)`!

### Naming Rules

-   Intermediate names must be alphanumeric (plus underscores)
-   They cannot conflict with:
    -   Global constants
    -   Function names
    -   Aliases
    -   Other intermediates in the same scope

### Complex Example

```scheme
//display
# Compute quadratic formula
discriminant = (- (* b b) (* 4 (* a c)))
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
//dupe true
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
//enddef

//display
//ret (horner x 2 -3 1 5)
```

### Example 2: Fibonacci with Loops

```scheme
//global N 10

//display
//repeat N i
  //local fib (if (< i 2)
                 i
                 (+ ans ans))  # This is kind of janky but it works
  //ret fib
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
//enddef

//def magnitude $x $y
//ret (** (+ (* $x $x) (* $y $y)) 0.5)
//enddef

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
//enddef

//def tanh_approx $x
//ret (- (* 2 (sigmoid (* 2 $x))) 1)
//enddef

//def relu $x
//ret (max0 $x)
//enddef

//def gelu $x
  sig = (sigmoid (* 1.702 $x))
  //ret (* $x sig)
//enddef

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
//enddef

//def my_function $x
//ret (* $x (* $x $x))  # x³
//enddef

//display
//ret (derivative my_function x)  # Should be ~3x²
```

### Example 6: Parametric Curves

```scheme
//global N 50

//display
//repeat N i
  //local t (/ i N)
  //local x_pos (cos (* 2 (* pi t)))
  //local y_pos (sin (* 2 (* pi t)))
  //ret (+ (* x_pos x_pos) (* y_pos y_pos))  # Should be 1 (circle!)
//endrepeat
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
| `//enddef without //def`                  | Mismatched tags        | Check structure           |

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
//store result1
//ret (complex_computation x)

//display
//ret (* ans 2)  # Might not be what you expect!

# Better
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
  //ret (* i i)
//endrepeat
```

### 6. Break Complex Functions into Smaller Ones

```scheme
# Bad: monolithic function
//def neural_network $x
//ret (... 50 lines of nested operations ...)
//enddef

# Good: modular design
//def sigmoid $x
//ret (/ 1 (+ 1 (** e (* -1 $x))))
//enddef

//def layer $x $w $b
//ret (sigmoid (+ (* $w $x) $b))
//enddef

//def neural_network $x
  h1 = (layer $x 0.5 0.1)
  h2 = (layer h1 0.3 -0.2)
  //ret (layer h2 0.8 0.0)
//enddef
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
                   "display" | "store" | "ret" | "simplify" | "dupe" |
                   "def" | "enddef" | "repeat" | "endrepeat" | "local"

code_block     ::= output_directive intermediate* ret_directive
output_directive ::= "//display" | "//store" var_name
intermediate   ::= identifier "=" expression newline
ret_directive  ::= "//ret" expression newline?

function       ::= "//def" func_name param* newline
                   intermediate* ret_directive newline
                   "//enddef"

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

## Migration Guide from v0.4

If you're migrating from v0.4, here are the key changes:

### What's New

1. **Loops**: `//repeat` and `//endrepeat` for iteration
2. **Locals**: `//local` for loop-scoped variables
3. **Intermediates**: Named subexpressions in code blocks
4. **Flexible constants**: `//global` can appear anywhere (renamed from `//const`)
5. **Aliases**: `//alias` for alternative variable names
6. **Return directive**: `//ret` explicitly marks the return value
7. **Built-in operations**: `mod`, `frac`, `nat` are now first-class
8. **Better optimization**: Improved duplicate detection

### What Changed

1. `//const` → `//global` (more accurate name)
2. Bare expressions → must use `//ret` in functions and code blocks
3. `//then` → removed (no longer needed with `//ret`)
4. Constants can be defined anywhere, not just at the start

### Example Migration

**v0.4 code:**

```scheme
//precision 6
//const RADIUS 5

//def area $r
(* pi (* $r $r))
//enddef

//display
(area RADIUS)
//then
```

**v2.0 code:**

```scheme
//precision 6
//global RADIUS 5

//def area $r
//ret (* pi (* $r $r))
//enddef

//display
//ret (area RADIUS)
```

---

## Contributing

Paramath is an evolving language. Contributions and feedback are welcome!

**Version**: 2.0  
**License**: MIT  
**GitHub**: https://github.com/CastyLoz17/paramath

---

_Last updated: December 2025_
