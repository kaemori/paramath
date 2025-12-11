# Paramath Language Specification v2.0

_a differentiable programming language for continuous computation_

---

## table of contents

1. [introduction](#introduction)
2. [syntax overview](#syntax-overview)
3. [data types](#data-types)
4. [operators and functions](#operators-and-functions)
5. [program structure](#program-structure)
6. [pragmas and directives](#pragmas-and-directives)
7. [function definitions](#function-definitions)
8. [lambda expressions](#lambda-expressions)
9. [loops and iteration](#loops-and-iteration)
10. [intermediates and code blocks](#intermediates-and-code-blocks)
11. [advanced features](#advanced-features)
12. [complete examples](#complete-examples)
13. [error reference](#error-reference)
14. [best practices](#best-practices)

---

## introduction

**paramath** is a domain-specific language designed for applications that evaluate mathematical statements (more specifically, mathematical functions in the pre-calculus domain) instead of normal machine code. to do this, paramath uses smooth and continuous approximations of logical operations making all computations differentiable.

> "why the name **_paramath_**?"
> <br>the word "paramath" actually comes from the portmanteau of "parenthesis" and "mathematics".

### key features

-   **differentiable logic**: all operations remain continuous
-   **s-expression syntax**: clean, unambiguous structure
-   **automatic optimization**: duplicate detection and subexpression extraction
-   **loop unrolling**: compile-time iteration for performance
-   **flexible output**: display or store results in variables

---

## syntax overview

### basic structure

paramath uses **s-expressions** (symbolic expressions) similar to lisp:

```scheme
(operator operand1 operand2 ...)
```

### comments

comments start with `#` and continue to end of line:

```scheme
# this is a comment
(+ 2 3)  # this adds 2 and 3
```

### variables

reserved variable names (case-insensitive):

-   single letters: `a`, `b`, `c`, `d`, `e`, `f`, `x`, `y`, `m`
-   special: `ans` (previous result), `pi`, `e`

> **note**: `ans` stores the result of the last computed expression. when using `//store X`, the value goes to both `X` and `ans`. **however**, when `//dupe true` is enabled (default), the compiler may create intermediate variables that make `ans` behave unpredictably - prefer using named intermediates instead!

---

## data types

### numeric literals

```scheme
42        # integer
3.14159   # float
-17       # negative integer
2.5e-3    # scientific notation (preferred for small numbers!)
```

### constants

define constants using `//global` anywhere in your program:

```scheme
//global MEANING 42
//global GOLDEN_RATIO (/ (+ 1 (** 5 0.5)) 2)
```

constants can reference other previously defined constants:

```scheme
//global RADIUS 5
//global AREA (* pi (* RADIUS RADIUS))
```

you can also set precision for individual constants:

```scheme
//precision 4
//global PI_APPROX pi  # evaluates pi to 4 decimal places

//precision 10
//global E_PRECISE e   # evaluates e to 10 decimal places
```

### aliases

create alternative names for variables:

```scheme
//alias input x        # "input" now refers to x
//alias output y       # "output" now refers to y
//alias theta a        # "theta" now refers to a

//display
//ret (sin theta)      # uses variable a
```

---

## operators and functions

### basic arithmetic

| operator | description    | example          |
| -------- | -------------- | ---------------- |
| `+`      | addition       | `(+ 2 3)` → 5    |
| `-`      | subtraction    | `(- 10 3)` → 7   |
| `*`      | multiplication | `(* 4 5)` → 20   |
| `/`      | division       | `(/ 10 2)` → 5   |
| `**`     | exponentiation | `(** 2 8)` → 256 |

**note**: `+` and `*` support multiple arguments and are right-associative:

```scheme
(+ 1 2 3 4)  # expands to (+ 1 (+ 2 (+ 3 4)))
```

### trigonometric functions

| function | description     |
| -------- | --------------- |
| `sin`    | sine            |
| `cos`    | cosine          |
| `tan`    | tangent         |
| `arcsin` | inverse sine    |
| `arccos` | inverse cosine  |
| `arctan` | inverse tangent |
| `abs`    | absolute value  |

example:

```scheme
(sin (* pi (/ x 2)))  # sin(πx/2)
```

### logical operations (differentiable)

all logical operations are continuous approximations that depend on epsilon:

| operation     | syntax     | description                       |
| ------------- | ---------- | --------------------------------- |
| equality      | `(== a b)` | returns ~1 when a≈b, ~0 otherwise |
| equal to zero | `(=0 a)`   | returns ~1 when a≈0, ~0 otherwise |
| greater than  | `(> a b)`  | returns ~1 when a>b, ~0 otherwise |
| greater/equal | `(>= a b)` | returns ~1 when a≥b, ~0 otherwise |
| less than     | `(< a b)`  | returns ~1 when a<b, ~0 otherwise |
| less/equal    | `(<= a b)` | returns ~1 when a≤b, ~0 otherwise |
| logical not   | `(! a)`    | returns 1-a                       |

> **tip**: the smaller your epsilon, the sharper these comparisons become. recommended: `1e-99` for maximum sharpness.

### mathematical operations

| operation | syntax      | description                      |
| --------- | ----------- | -------------------------------- |
| sign      | `(sign a)`  | returns -1, 0, or 1 (continuous) |
| max       | `(max a b)` | maximum of a and b               |
| min       | `(min a b)` | minimum of a and b               |
| max0      | `(max0 a)`  | maximum of a and 0               |
| min0      | `(min0 a)`  | minimum of a and 0               |
| modulo    | `(mod a b)` | a modulo b (differentiable!)     |
| fraction  | `(frac a)`  | fractional part of a             |
| natural   | `(nat a)`   | rounds a to nearest integer      |

### conditional operations

```scheme
(if condition then_value else_value)
```

example:

```scheme
(if (== 0 (mod x 2))
    (/ x 2)
    (+ (* 3 x) 1))  # collatz conjecture!
```

---

## program structure

a paramath program consists of:

1. **configuration** (optional): precision, epsilon, optimization settings
2. **constants** (optional): named constant definitions
3. **aliases** (optional): alternative variable names
4. **functions** (optional): user-defined functions
5. **code blocks**: computation blocks with output directives

**note**: unlike v0.4, configurations and constants can appear anywhere! they only affect code that comes after them.

### minimal program

```scheme
//display
//ret + 2 2
```

**note**: the final surrounding parenthesis is not necessarily needed if you're using a single line string, but if any error arises, surrounding the expression with parenthesis usually resolves it.

### complete program structure

```scheme
# configuration
//precision 6
//epsilon 1e-99
//simplify true
//dupe true

# constants
//global RADIUS 5

# aliases
//alias r x

# functions
//def circle_area $r
//ret (* pi (* $r $r))
//enddef

# computation
//display
//ret (circle_area RADIUS)
```

---

## pragmas and directives

pragmas are special commands that start with `//`.

### configuration pragmas

can appear anywhere and affect subsequent code:

#### `//precision`

sets decimal precision for constant evaluation:

```scheme
//precision 4
//global PI_LOW pi  # evaluates to 4 decimals

//precision 10
//global PI_HIGH pi  # evaluates to 10 decimals
```

#### `//epsilon`

sets the epsilon parameter for logical operations:

```scheme
//epsilon 1e-99  # sharp comparisons
//epsilon 1e-50  # slightly softer
```

smaller epsilon = sharper decision boundaries. recommended: `1e-99` for most use cases.

#### `//simplify`

enables/disables compile-time simplification of literal expressions:

```scheme
//simplify true   # default: simplifies (+ 2 3) to 5 at compile time
//simplify false  # keeps all expressions as-is
```

#### `//dupe`

enables/disables duplicate subexpression detection:

```scheme
//dupe true   # default: extracts repeated subexpressions
//dupe false  # keeps expression as-is, even with duplicates
```

> **warning**: when `//dupe true`, the compiler may create intermediate results automatically. this makes `ans` less predictable - use named variables instead!

### output directives

every code block must have an output directive:

#### `//display`

output result to console:

```scheme
//display
//ret (+ 2 3)
```

#### `//store`

store result in a variable:

```scheme
//store x
//ret (+ 2 3)

//display
//ret (* x 2)  # uses stored value
```

valid variable names: `a`, `b`, `c`, `d`, `e`, `f`, `x`, `y`, `m`

---

## function definitions

### syntax

```scheme
//def function_name $param1 $param2 ...
//ret (function body)
//enddef
```

-   function names are case-insensitive
-   parameters must start with `$`
-   body starts with `//ret` (return directive)

### examples

```scheme
//def square $x
//ret (* $x $x)
//enddef

//def distance $x1 $y1 $x2 $y2
//ret (** (+ (** (- $x2 $x1) 2) (** (- $y2 $y1) 2)) 0.5)
//enddef

//display
//ret (distance 0 0 3 4)  # returns 5.0
```

### recursion

functions can call themselves:

```scheme
//def factorial $n
//ret (if (<= $n 1)
        1
        (* $n (factorial (- $n 1))))
//enddef
```

---

## lambda expressions

anonymous functions for one-time use.

### syntax

```scheme
((lambda ($param1 $param2 ...) body) arg1 arg2 ...)
```

### examples

```scheme
# simple lambda
((lambda ($x) (* $x $x)) 5)  # returns 25

# multiple parameters
((lambda ($x $y) (+ $x $y)) 3 4)  # returns 7

# nested lambdas
((lambda ($f $x) ($f ($f $x)))
 (lambda ($y) (* $y 2))
 5)  # returns 20 (doubles 5 twice)
```

### lambda rules

-   must be immediately applied (no storing lambdas)
-   parameters must start with `$`
-   body is a single expression (no `//ret` needed!)

---

## loops and iteration

### basic loop syntax

```scheme
//repeat count_expression
  # loop body
//endrepeat
```

loops are **unrolled at compile time** - the loop body is literally copied `count` times with substitutions.

### simple loop

```scheme
//global SIZE 3

//display
//repeat SIZE
  //ret (+ x 1)
//endrepeat
```

this generates 3 separate expressions, each computing `(+ x 1)`.

### loops with iterator variable

```scheme
//repeat 5 i
  //ret (* i i)
//endrepeat
```

the iterator `i` takes values 0, 1, 2, 3, 4 in each unrolled iteration.

### local variables in loops

use `//local` to define variables that only exist within the loop:

```scheme
//display
//repeat 3 i
  //local square (* i i)
  //ret (+ square 1)
//endrepeat
```

each iteration gets its own `square` variable with the correct value.

### nested loops

loops can be nested:

```scheme
//display
//repeat 3 i
  //repeat 3 j
    //ret (+ (* i 10) j)
  //endrepeat
//endrepeat
```

### advanced loop example

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

## intermediates and code blocks

### intermediate assignments

instead of nested parentheses hell, you can break expressions into named parts:

```scheme
//display
a = (* x x)
b = (* y y)
c = (+ a b)
//ret (** c 0.5)
```

this is **much** cleaner than `(** (+ (* x x) (* y y)) 0.5)`!

### naming rules

-   intermediate names must be alphanumeric (plus underscores)
-   they cannot conflict with:
    -   global constants
    -   function names
    -   aliases
    -   other intermediates in the same scope

### complex example

```scheme
//display
# compute quadratic formula
discriminant = (- (* b b) (* 4 (* a c)))
sqrt_disc = (** discriminant 0.5)
numerator = (+ (* -1 b) sqrt_disc)
denominator = (* 2 a)
//ret (/ numerator denominator)
```

### automatic expansion

the compiler automatically expands intermediates into the final expression. you write clean code, it generates optimized math!

---

## advanced features

### automatic duplicate detection

when `//dupe true` (default), the compiler finds repeated subexpressions and extracts them automatically:

```scheme
//dupe true
//display
//ret (+ (* (+ x 1) (+ x 1)) (* (+ x 1) (+ x 1)))
```

the compiler recognizes `(+ x 1)` appears 4 times and may create an intermediate for it.

### epsilon in expressions

access the current epsilon value using `ε` or `epsilon`:

```scheme
//epsilon 0.01
//display
//ret (/ x (+ x ε))  # uses current epsilon value
```

### expression length optimization

the compiler tracks expression length and tries to minimize it:

-   extracts beneficial duplicates (saves space)
-   flattens associative operations
-   applies identity simplifications (e.g., `x * 1` → `x`)

---

## complete examples

### example 1: polynomial evaluation with horner's method

```scheme
//epsilon 1e-99

# evaluate polynomial: ax^3 + bx^2 + cx + d
//def horner $x $a $b $c $d
  term1 = (* $a $x)
  term2 = (+ term1 $b)
  term3 = (* term2 $x)
  term4 = (+ term3 $c)
  term5 = (* term4 $x)
  //ret (+ term5 $d)
//enddef

//display
//ret (horner x 2 (-3) 1 5)
```

### example 2: fibonacci with loops

```scheme
//global N 10

//display
//repeat N i
  //local fib (if (< i 2)
                 i
                 (+ ans ans))  # this is kinda jank but it works
  //ret fib
//endrepeat
```

### example 3: vector operations

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
//ret (/ dot (* vec1_mag vec2_mag))  # cosine of angle
```

### example 4: smooth activation functions

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

### example 5: numerical differentiation

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
//ret (* $x (* $x $x))  # x^3
//enddef

//display
//ret (derivative my_function x)  # should be ~3x^2
```

### example 6: parametric curves

```scheme
//global N 50

//display
//repeat N i
  //local t (/ i N)
  //local x_pos (cos (* 2 (* pi t)))
  //local y_pos (sin (* 2 (* pi t)))
  //ret (+ (* x_pos x_pos) (* y_pos y_pos))  # should be 1 (circle!)
//endrepeat
```

---

## error reference

### parser errors

| error                                     | cause                                  | solution                                 |
| ----------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `unexpected end of tokens`                | missing closing parenthesis            | add `)`                                  |
| `unexpected closing parenthesis`          | extra `)` or missing `(`               | check balance                            |
| `unknown identifier 'X'`                  | undefined variable/constant            | define with `//global` or check spelling |
| `unknown operation 'X'`                   | undefined function                     | define with `//def` or check spelling    |
| `incomplete expression at line X`         | unbalanced parentheses in intermediate | check your assignment                    |
| `expression without assignment at line X` | forgot `=` in intermediate             | use `var_name = expr` format             |

### block errors

| error                                | cause                                | solution             |
| ------------------------------------ | ------------------------------------ | -------------------- |
| `no output mode before codeblock`    | missing `//display` or `//store`     | add output directive |
| `codeblock has no //ret directive`   | missing `//ret`                      | add `//ret expr`     |
| `incomplete //ret expression`        | unbalanced parentheses after `//ret` | check parentheses    |
| `'X' conflicts with global constant` | naming conflict                      | use different name   |
| `'X' conflicts with function name`   | naming conflict                      | use different name   |

### loop errors

| error                                            | cause                  | solution                             |
| ------------------------------------------------ | ---------------------- | ------------------------------------ |
| `//repeat without matching //endrepeat`          | missing `//endrepeat`  | add `//endrepeat`                    |
| `//endrepeat without matching //repeat`          | extra `//endrepeat`    | remove it                            |
| `repeat range must be non-negative`              | negative loop count    | use positive number                  |
| `//local can only be used inside //repeat loops` | `//local` outside loop | move inside loop or use intermediate |
| `'X' already defined in this loop`               | duplicate local name   | use different name                   |

### function errors

| error                                     | cause                  | solution                  |
| ----------------------------------------- | ---------------------- | ------------------------- |
| `function 'X' expects N arguments, got M` | wrong argument count   | check function definition |
| `function parameters must start with $`   | invalid parameter name | use `$param` format       |
| `//enddef without //def`                  | mismatched tags        | check structure           |

### lambda errors

| error                                 | cause                | solution                    |
| ------------------------------------- | -------------------- | --------------------------- |
| `lambda must be immediately applied`  | standalone lambda    | wrap: `((lambda ...) args)` |
| `lambda expects N arguments, got M`   | wrong argument count | check parameters            |
| `lambda parameters must start with $` | invalid parameter    | use `$param` format         |

---

## best practices

### 1. use small epsilon values

```scheme
//epsilon 1e-99  # recommended for sharp comparisons
```

**avoid** epsilon values larger than `1e-50` unless you specifically need smooth transitions.

### 2. prefer intermediates over deeply nested expressions

```scheme
# bad
//ret (/ (+ (** (- x2 x1) 2) (** (- y2 y1) 2)) (+ (abs (- x2 x1)) (abs (- y2 y1))))

# good
dx = (- x2 x1)
dy = (- y2 y1)
euclidean = (** (+ (* dx dx) (* dy dy)) 0.5)
manhattan = (+ (abs dx) (abs dy))
//ret (/ euclidean manhattan)
```

### 3. use aliases for semantic clarity

```scheme
//alias input_voltage x
//alias output_current y
//alias resistance a

//display
//ret (/ input_voltage resistance)  # ohm's law
```

### 4. avoid relying on `ans` with dupe detection

when `//dupe true`, the compiler may create hidden intermediates that clobber `ans`. instead:

```scheme
# questionable
//store result1
//ret (complex_computation x)

//display
//ret (* ans 2)  # might not be what you expect!

# better
//store result1
//ret (complex_computation x)

//display
//ret (* result1 2)  # explicit and clear
```

### 5. use loops for repeated patterns

```scheme
# bad: copy-paste hell
//display
//ret (* 0 0)
//display
//ret (* 1 1)
//display
//ret (* 2 2)
# ... 97 more times

# good: loop
//display
//repeat 100 i
  //ret (* i i)
//endrepeat
```

### 6. break complex functions into smaller ones

```scheme
# bad: monolithic function
//def neural_network $x
//ret (... 50 lines of nested operations ...)
//enddef

# good: modular design
//def sigmoid $x
//ret (/ 1 (+ 1 (** e (* -1 $x))))
//enddef

//def layer $x $w $b
//ret (sigmoid (+ (* $w $x) $b))
//enddef

//def neural_network $x
  h1 = (layer $x 0.5 0.1)
  h2 = (layer h1 0.3 (-0.2))
  //ret (layer h2 0.8 0.0)
//enddef
```

### 7. comment your epsilon choices

```scheme
# sharp comparison for exact equality check
//epsilon 1e-99
//display
//ret (== x y)

# smooth transition for gradient-based optimization
//epsilon 1e-10
//display
//ret (if (> x threshold) high_val low_val)
```

### 8. use meaningful constant names

```scheme
# bad
//global C1 299792458
//global C2 6.62607015e-34

# good
//global SPEED_OF_LIGHT 299792458
//global PLANCK_CONSTANT 6.62607015e-34
```

---

## language grammar (ebnf)

```ebnf
program        ::= (pragma | function | code_block)*

pragma         ::= "//" pragma_name (pragma_args)?
pragma_name    ::= "precision" | "epsilon" | "global" | "alias" |
                   "display" | "store" | "ret" | "simplify" | "dupe" |
                   "def" | "enddef" | "repeat" | "endrepeat" | "local"

code_block     ::= output_directive (intermediate)* ret_directive
output_directive ::= "//display" | "//store" var_name
intermediate   ::= identifier "=" expression newline
ret_directive  ::= "//ret" expression (newline)?

function       ::= "//def" func_name param* newline
                   (intermediate)* ret_directive newline
                   "//enddef"

loop           ::= "//repeat" expression (identifier)? newline
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
number         ::= ["-"] digit+ ["." digit+] [("e"|"E") ["-"|"+"] digit+]
identifier     ::= (alpha | "_") (alpha | digit | "_")*
```

---

## migration guide from v0.4

if you're migrating from v0.4, here are the key changes:

### what's new

1. **loops**: `//repeat` and `//endrepeat` for iteration
2. **locals**: `//local` for loop-scoped variables
3. **intermediates**: named subexpressions in code blocks
4. **flexible constants**: `//global` can appear anywhere (renamed from `//const`)
5. **aliases**: `//alias` for alternative variable names
6. **return directive**: `//ret` explicitly marks the return value
7. **built-in operations**: `mod`, `frac`, `nat` are now first-class
8. **better optimization**: improved duplicate detection

### what changed

1. `//const` → `//global` (more accurate name)
2. bare expressions → must use `//ret` in functions and code blocks
3. `//then` → removed (no longer needed with `//ret`)
4. constants can be defined anywhere, not just at the start

### example migration

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

## contributing

paramath is an evolving language. contributions and feedback are welcome!

**version**: 2.0  
**license**: MIT  
**github**: _add this link later ig_

---

_last updated: december 2025_
