import json
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy
import re


setflags = (
    lambda logfile, verbose, debug: (logfile << 2) | (verbose << 1) | (debug << 0)
)
flags = 0b000

VERBOSE = bool(flags & (1 << 0))
DEBUG = bool(flags & (1 << 1))
LOGFILE = bool(flags & (1 << 2))

VAR_NAMES = list("abcdefxym") + ["ans", "pi", "e"]
BUILTIN_FUNCS = ["abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan"]
BASIC_OPS = {"+", "-", "*", "/", "**"} | set(BUILTIN_FUNCS)


OP_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}
OPERATION_EXPANSIONS = {
    "==": lambda a, b, eps: [
        "/",
        ["abs", ["-", a, b]],
        ["+", ["abs", ["-", a, b]], eps],
    ],
    "=0": lambda a, eps: ["/", ["abs", a], ["+", ["abs", a], eps]],
    ">": lambda a, b, eps: [
        "/",
        ["+", ["-", a, b], ["abs", ["-", a, b]]],
        ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
    ],
    ">=": lambda a, b, eps: [
        "/",
        ["-", ["+", ["-", a, b], ["abs", ["-", a, b]]], eps],
        ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
    ],
    "<": lambda a, b, eps: [
        "/",
        ["+", ["-", b, a], ["abs", ["-", b, a]]],
        ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
    ],
    "<=": lambda a, b, eps: [
        "/",
        ["-", ["+", ["-", b, a], ["abs", ["-", b, a]]], eps],
        ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
    ],
    "!": lambda a: ["-", 1, a],
    "sign": lambda a, eps: ["/", a, ["+", ["abs", a], eps]],
    "max": lambda a, b: ["/", ["+", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "max0": lambda a: ["/", ["+", a, ["abs", a]], 2],
    "min": lambda a, b: ["/", ["-", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "min0": lambda a: ["/", ["-", a, ["abs", a]], 2],
    "if": lambda cond, then_val, else_val: [
        "+",
        ["*", then_val, cond],
        ["*", else_val, ["-", 1, cond]],
    ],
    "mod": lambda a, b: [
        "-",
        ["/", b, 2],
        [
            "/",
            ["*", b, ["arctan", ["/", 1, ["tan", ["*", ["*", "pi", a], ["/", 1, b]]]]]],
            "pi",
        ],
    ],
    "frac": lambda a: [
        "-",
        ["-", a, 0.5],
        ["/", ["arctan", ["tan", ["*", "pi", ["-", a, 0.5]]]], "pi"],
    ],
    "nat": lambda a: [
        "+",
        a,
        ["/", ["arctan", ["tan", ["*", "pi", ["-", a, 0.5]]]], "pi"],
        0.5,
    ],
}


_caches = {
    "simplify": {},
    "length": {},
    "generate": {},
    "try_simplify": {},
    "sympy": {},
    "substitute": {},
}


def clear_all_caches():
    for cache in _caches.values():
        cache.clear()
    debug_print("cleared all caches")


def cached(cache_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = _caches[cache_name]
            key = json.dumps([args, kwargs], sort_keys=True, default=str)
            if key in cache:
                debug_print(f"cache hit for {func.__name__}")
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            debug_print(f"cache miss for {func.__name__}, storing result")
            return result

        return wrapper

    return decorator


def _log_message(prefix: str, message: str, max_len: int = 120):
    original_len = len(message)
    truncated = message[:max_len]
    log_line = f"{prefix} {original_len}] {truncated}"
    print(log_line)
    if LOGFILE:
        with open("logfile.txt", "a") as f:
            f.write(log_line + "\n")


def debug_print(message: str):
    if DEBUG:
        _log_message("[DEBUG", message)


def verbose_print(message: str):
    if VERBOSE:
        _log_message("[VERBOSE", message)


class ParsePhase(Enum):
    config = auto()
    constants = auto()
    functions = auto()
    output = auto()
    code = auto()
    done = auto()


@dataclass
class Function:
    name: str
    params: List[str]
    body: Any
    line_num: int


@dataclass
class LoopContext:
    range_val: int
    var_name: Optional[str]
    start_line: int
    body_lines: List[Tuple[int, str]] = field(default_factory=list)
    locals_in_scope: set = field(default_factory=set)


@dataclass
class ProgramConfig:
    constants: Dict[str, float] = field(default_factory=dict)
    functions: Dict[str, Function] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    precision: Optional[int] = 10
    epsilon: Union[float, str] = 10e-99
    current_epsilon: Union[float, str] = None
    current_precision: Optional[int] = None
    output_mode: Optional[Tuple[str, Optional[str]]] = None
    simplify_literals: bool = True
    current_simplify: bool = None
    use_sympy: bool = False
    current_sympy: bool = None
    detect_duplicates: bool = True
    current_dupe: bool = None
    dupe_min_savings: int = -999
    current_dupe_min_savings: int = None
    loop_stack: List[LoopContext] = field(default_factory=list)
    var_names: List[str] = field(
        default_factory=lambda: list("abcdefxym") + ["pi", "e"]
    )

    def __post_init__(self):
        if self.current_epsilon is None:
            self.current_epsilon = self.epsilon
        if self.current_precision is None:
            self.current_precision = self.precision
        if self.current_sympy is None:
            self.current_sympy = self.use_sympy
        if self.current_simplify is None:
            self.current_simplify = self.simplify_literals
        if self.current_dupe is None:
            self.current_dupe = self.detect_duplicates
        if self.current_dupe_min_savings is None:
            self.current_dupe_min_savings = self.dupe_min_savings


@dataclass
class CodeBlock:
    intermediates: Dict[str, str] = field(default_factory=dict)
    ret_expr: Optional[str] = None
    line_start: int = 0
    line_end: int = 0


@dataclass
class Instruction:
    ast: Any
    output: Optional[Tuple[str, Optional[str]]]
    epsilon: Union[float, str]
    precision: Optional[int]
    simplify: bool
    sympy: bool
    dupe: bool
    dupe_min_savings: int
    line_start: int
    line_end: int


class ParserError(Exception):
    def __init__(self, message: str, line_num: Optional[int] = None):
        super().__init__(f"line {line_num}: {message}" if line_num else message)
        self.line_num = line_num


def num(val):
    for caster in (int, float):
        try:
            return caster(val)
        except ValueError:
            pass
    raise ValueError(f"cannot turn {val!r} into a number")


def tokenize(code: str) -> List[str]:
    last_code = None
    while last_code != code:
        last_code = code
        code = code.replace("  ", " ")

    code = code.replace("**", " ** ")
    for op in [" + ", " - ", " * ", " / "]:
        code = code.replace(op, f" {op} ")
    code = code.replace("(", " ( ").replace(")", " ) ")
    tokens = code.split()
    debug_print(f"tokenized into {len(tokens)} tokens: {tokens[:10]}...")
    return tokens


def all_sublists(expr):
    if isinstance(expr, list):
        yield expr
        for item in expr:
            yield from all_sublists(item)


def replace_lists(target, pattern):
    if target == pattern:
        return "ANS"
    if isinstance(target, list):
        return [replace_lists(x, pattern) for x in target]
    return target


def parse_intermediate_assignment(line: str) -> Optional[Tuple[str, str]]:
    if "=" not in line:
        return None
    parts = line.split("=", 1)
    if len(parts) != 2:
        return None
    var_name = parts[0].strip()
    if not var_name.replace("_", "").isalnum():
        return None
    debug_print(f"parsed intermediate assignment: {var_name}")
    return (var_name, parts[1].strip())


def check_naming_conflicts(
    name: str, config: ProgramConfig, line_num: int, context: str = ""
):
    name_lower = name.lower()
    conflicts = [
        (name in config.constants, "global constant"),
        (name_lower in config.aliases, "alias"),
        (name_lower in config.functions, "function name"),
    ]

    for has_conflict, conflict_type in conflicts:
        if has_conflict:
            raise ParserError(
                f"{context}'{name}' conflicts with {conflict_type}", line_num
            )

    if config.loop_stack:
        current_loop = config.loop_stack[-1]
        if name in current_loop.locals_in_scope:
            raise ParserError(
                f"{context}'{name}' already defined in this loop", line_num
            )

    debug_print(f"no naming conflicts for '{name}'")


@cached("substitute")
def substitute_params(ast: Any, params: List[str], args: List[Any]) -> Any:
    if isinstance(ast, (int, float)):
        return ast
    if isinstance(ast, str):
        return deepcopy(args[params.index(ast)]) if ast in params else ast
    if isinstance(ast, list):
        return [substitute_params(node, params, args) for node in ast]
    return ast


def compile_value(
    expr_str: str,
    constants: Dict[str, float],
    line_num: int,
    config: ProgramConfig,
    loop_vars: Optional[Dict[str, Any]] = None,
) -> float:
    expr_str = expr_str.strip()
    verbose_print(f"compiling value: {expr_str[:50]}...")

    try:
        result = num(expr_str)
        debug_print(f"direct numeric conversion: {result}")
        return result
    except ValueError:
        pass

    try:
        loop_vars = loop_vars or {}
        import builtins

        eval_namespace = {
            "__builtins__": builtins.__dict__,
            "consts": type("Consts", (), constants)(),
            "local": type("Local", (), loop_vars)(),
            **constants,
        }
        result = num(eval(expr_str, eval_namespace))
        debug_print(f"eval'd to: {result}")
        return result
    except Exception as e:
        debug_print(f"eval failed: {e}, trying ast parse")

    try:
        tokens = tokenize(expr_str)
        if not tokens:
            raise ParserError("empty expression", line_num)
        if tokens[0] != "(":
            if len(tokens) > 1:
                tokens = ["("] + tokens + [")"]
        ast = parse_tokens(tokens, line_num)
        simplified = simplify_ast(
            ast,
            config.current_epsilon,
            constants,
            config.functions,
            config.aliases,
            line_num,
        )
        expr = generate_expression(simplified, line_num)
        result = num(eval(expr))
        debug_print(f"ast parse succeeded: {result}")
        return result
    except Exception as e:
        raise ParserError(f"failed to compile value: {e}", line_num)


def collect_codeblock(
    expanded_code: List[Tuple[int, str]], start_idx: int, config: ProgramConfig
) -> Tuple[CodeBlock, int]:
    block = CodeBlock(line_start=expanded_code[start_idx][0])
    current_expr_lines = []
    i = start_idx
    in_ret_expr = False
    ret_start_line = None

    verbose_print(f"collecting codeblock starting at line {block.line_start}")

    while i < len(expanded_code):
        line_num, line = expanded_code[i]

        if line.lower().startswith("//ret"):
            if current_expr_lines:
                raise ParserError(
                    "found //ret but previous expression has no variable assignment",
                    line_num,
                )
            parts = line.split(None, 1)
            if len(parts) >= 2:
                ret_expr = parts[1]
                paren_depth = sum(
                    1 if c == "(" else -1 if c == ")" else 0 for c in ret_expr
                )
                if paren_depth == 0:
                    ret_expr_stripped = ret_expr.strip()
                    if not ret_expr_stripped.startswith("("):
                        tokens_test = ret_expr_stripped.split()
                        if len(tokens_test) > 1:
                            ret_expr = f"({ret_expr})"
                    block.ret_expr = ret_expr
                    block.line_end = line_num
                    debug_print(f"collected complete ret expr: {ret_expr[:50]}...")
                    return block, i - start_idx + 1
                else:
                    current_expr_lines.append(ret_expr)
                    in_ret_expr = True
                    ret_start_line = line_num
            else:
                in_ret_expr = True
                ret_start_line = line_num
            i += 1
            continue

        if line.lower().startswith("//"):
            if current_expr_lines:
                raise ParserError(
                    f"incomplete intermediate assignment at line {line_num}", line_num
                )
            break

        current_expr_lines.append(line)
        full_expr = " ".join(current_expr_lines)
        paren_depth = sum(1 if c == "(" else -1 if c == ")" else 0 for c in full_expr)

        if paren_depth == 0:
            if in_ret_expr:
                block.ret_expr = full_expr
                block.line_end = line_num
                debug_print(f"collected multiline ret expr")
                return block, i - start_idx + 1

            assignment = parse_intermediate_assignment(full_expr)
            if assignment:
                var_name, expr = assignment
                check_naming_conflicts(var_name, config, line_num, "intermediate ")
                block.intermediates[var_name] = expr
                current_expr_lines.clear()
                debug_print(f"collected intermediate: {var_name} = {expr[:30]}...")
            else:
                raise ParserError(
                    f"expression without assignment at line {line_num}", line_num
                )
        i += 1

    if current_expr_lines:
        if in_ret_expr:
            raise ParserError(
                f"incomplete //ret expression starting at line {ret_start_line}",
                expanded_code[i - 1][0],
            )
        raise ParserError(
            "incomplete expression at end of codeblock", expanded_code[i - 1][0]
        )
    if block.ret_expr is None:
        raise ParserError("codeblock has no //ret directive", block.line_start)

    return block, i - start_idx


def expand_intermediates(
    ret_expr: str, intermediates: Dict[str, str], line_num: int
) -> str:
    result = ret_expr
    max_iterations = 100
    iteration = 0

    verbose_print(f"expanding {len(intermediates)} intermediates")

    while iteration < max_iterations:
        changed = False
        for var_name, var_expr in intermediates.items():
            pattern = r"\b" + re.escape(var_name) + r"\b"
            if re.search(pattern, result):
                result = re.sub(pattern, f"({var_expr})", result)
                changed = True
                debug_print(f"iteration {iteration}: substituted {var_name}")
        if not changed:
            verbose_print(f"expansion complete after {iteration} iterations")
            break
        iteration += 1

    if iteration >= max_iterations:
        raise ParserError(
            "infinite loop detected in intermediate variable substitution", line_num
        )

    return result


def parse_tokens(tokens: List[str], line_num: int) -> Any:
    if not tokens:
        raise ParserError("unexpected end of tokens", line_num)

    token = tokens.pop(0)

    if token == "(":
        expr = []
        while tokens and tokens[0] != ")":
            expr.append(parse_tokens(tokens, line_num))
        if not tokens:
            raise ParserError("missing closing parenthesis ')'", line_num)
        tokens.pop(0)
        debug_print(f"parsed expression with {len(expr)} elements")
        return expr
    elif token == ")":
        raise ParserError("unexpected closing parenthesis ')'", line_num)
    else:
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token


def apply_identity_simplifications(op: str, operands: List[Any]) -> Optional[Any]:
    if op == "*":
        if 0 in operands:
            debug_print("simplified multiplication by zero")
            return 0
        if len(operands) == 2:
            if operands[0] == 1:
                return operands[1]
            if operands[1] == 1:
                return operands[0]

            if operands[0] == operands[1]:
                debug_print("simplified x * x via structural")
                return ["**", operands[0], 2]
    elif op == "+":
        if len(operands) == 2:
            if operands[0] == 0:
                return operands[1]
            if operands[1] == 0:
                return operands[0]

            if operands[0] == operands[1]:
                debug_print("simplified a + a = 2*a")
                return ["*", 2, operands[0]]
    elif op == "-":
        if len(operands) == 2:
            if operands[1] == 0:
                return operands[0]
            if operands[0] == operands[1]:
                debug_print("simplified x - x = 0")
                return 0
    elif op == "/":
        if len(operands) == 2:
            if operands[0] == 0:
                return 0
            if operands[1] == 1:
                return operands[0]

            if operands[0] == operands[1]:
                debug_print("simplified x / x = 1")
                return 1
    elif op == "**":
        if len(operands) == 2:
            if operands[1] == 0:
                return 1
            if operands[1] == 1:
                return operands[0]
            if operands[0] in [0, 1]:
                return operands[0]
    return None


def apply_structural_simplifications(op: str, operands: List[Any]) -> Optional[Any]:
    if op == "+":

        flat_operands = []
        for operand in operands:
            if isinstance(operand, list) and len(operand) >= 2 and operand[0] == "+":
                flat_operands.extend(operand[1:])
            else:
                flat_operands.append(operand)

        term_counts = {}
        for operand in flat_operands:
            key = json.dumps(operand, sort_keys=False)
            if key not in term_counts:
                term_counts[key] = {"count": 0, "term": operand}
            term_counts[key]["count"] += 1

        new_terms = []
        for data in term_counts.values():
            if data["count"] == 1:
                new_terms.append(data["term"])
            else:
                new_terms.append(["*", data["count"], data["term"]])

        if len(new_terms) == 0:
            return 0
        elif len(new_terms) == 1:
            return new_terms[0]
        elif len(new_terms) != len(flat_operands):

            debug_print(
                f"simplified repeated addition: {len(flat_operands)} -> {len(new_terms)} terms"
            )

            result = new_terms[-1]
            for term in reversed(new_terms[:-1]):
                result = ["+", term, result]
            return result

    elif op == "*":

        flat_operands = []
        for operand in operands:
            if isinstance(operand, list) and len(operand) >= 2 and operand[0] == "*":
                flat_operands.extend(operand[1:])
            else:
                flat_operands.append(operand)

        base_counts = {}
        numeric_product = 1

        for operand in flat_operands:

            if isinstance(operand, (int, float)):
                numeric_product *= operand
                continue

            if isinstance(operand, list) and len(operand) == 3 and operand[0] == "**":
                base = operand[1]
                power = operand[2]
                key = json.dumps(base, sort_keys=False)
                if key not in base_counts:
                    base_counts[key] = {"power": 0, "base": base}
                base_counts[key]["power"] += power
            else:
                key = json.dumps(operand, sort_keys=False)
                if key not in base_counts:
                    base_counts[key] = {"power": 0, "base": operand}
                base_counts[key]["power"] += 1

        new_factors = []

        if numeric_product != 1:
            if numeric_product == 0:
                return 0
            new_factors.append(numeric_product)

        for data in base_counts.values():
            if data["power"] == 0:
                continue
            elif data["power"] == 1:
                new_factors.append(data["base"])
            else:
                new_factors.append(["**", data["base"], data["power"]])

        if len(new_factors) == 0:
            return 1
        elif len(new_factors) == 1:
            return new_factors[0]
        elif len(new_factors) != len(flat_operands):

            debug_print(
                f"simplified repeated multiplication: {len(flat_operands)} -> {len(new_factors)} factors"
            )

            result = new_factors[-1]
            for factor in reversed(new_factors[:-1]):
                result = ["*", factor, result]
            return result

    return None


@cached("simplify")
def simplify_ast(
    ast: Any,
    epsilon: str,
    constants: Dict[str, float],
    functions: Dict[str, Function],
    aliases: Dict[str, str],
    line_num: int,
    var_names: List[str] = None,
    do_simplify: bool = True,
) -> Any:
    if var_names is None:
        var_names = list("abcdefxym") + ["ans", "pi", "e"]

    if isinstance(ast, (int, float)):
        return ast

    if isinstance(ast, str):
        ast_lower = ast.lower()
        if ast_lower in aliases:
            actual_var = aliases[ast_lower]
            if actual_var in constants:
                return constants[actual_var]
            if actual_var.lower() in var_names:
                return actual_var.lower()
            raise ParserError(
                f"alias '{ast}' points to unknown variable '{actual_var}'", line_num
            )
        if ast in constants:
            return constants[ast]
        if ast_lower in var_names:
            return ast_lower
        if ast == "ε" or ast_lower == "epsilon":
            return epsilon
        raise ParserError(f"unknown identifier '{ast}'", line_num)

    if isinstance(ast, list) and len(ast) == 1:
        return simplify_ast(
            ast[0],
            epsilon,
            constants,
            functions,
            aliases,
            line_num,
            var_names,
            do_simplify,
        )

    if not isinstance(ast, list) or len(ast) == 0:
        raise ParserError(f"invalid ast node: {ast}", line_num)

    if (
        isinstance(ast[0], list)
        and len(ast[0]) > 0
        and isinstance(ast[0][0], str)
        and ast[0][0].lower() == "lambda"
    ):
        lambda_expr = ast[0]
        if len(lambda_expr) < 3:
            raise ParserError("lambda requires parameter list and body", line_num)
        params = lambda_expr[1]
        if not isinstance(params, list):
            raise ParserError("lambda parameters must be a list", line_num)
        for param in params:
            if not isinstance(param, str) or not param.startswith("$"):
                raise ParserError(
                    f"lambda parameters must start with $ (got '{param}')", line_num
                )
        body = lambda_expr[2]
        args = ast[1:]
        if len(args) != len(params):
            raise ParserError(
                f"lambda expects {len(params)} arguments, got {len(args)}", line_num
            )
        substituted_body = substitute_params(body, params, args)
        return simplify_ast(
            substituted_body,
            epsilon,
            constants,
            functions,
            aliases,
            line_num,
            var_names,
            do_simplify,
        )

    if isinstance(ast[0], str) and ast[0].lower() == "lambda":
        raise ParserError(
            "lambda must be immediately applied (use ((lambda ...) args) syntax)",
            line_num,
        )

    op = ast[0]
    if isinstance(op, list):
        op = simplify_ast(
            op, epsilon, constants, functions, aliases, line_num, var_names, do_simplify
        )

    if isinstance(op, str):
        op_lower = op.lower()

        if op_lower in functions:
            func = functions[op_lower]
            args = ast[1:]
            if len(args) != len(func.params):
                raise ParserError(
                    f"function '{func.name}' expects {len(func.params)} arguments, got {len(args)}",
                    line_num,
                )
            substituted_body = substitute_params(func.body, func.params, args)
            return simplify_ast(
                substituted_body,
                epsilon,
                constants,
                functions,
                aliases,
                line_num,
                var_names,
                do_simplify,
            )

        if op_lower in BASIC_OPS:
            simplified_operands = [
                simplify_ast(
                    node,
                    epsilon,
                    constants,
                    functions,
                    aliases,
                    line_num,
                    var_names,
                    do_simplify,
                )
                for node in ast[1:]
            ]

            if do_simplify:
                identity_result = apply_identity_simplifications(
                    op_lower, simplified_operands
                )
                if identity_result is not None:
                    return simplify_ast(
                        identity_result,
                        epsilon,
                        constants,
                        functions,
                        aliases,
                        line_num,
                        var_names,
                        do_simplify,
                    )

                structural_result = apply_structural_simplifications(
                    op_lower, simplified_operands
                )
                if structural_result is not None:
                    return simplify_ast(
                        structural_result,
                        epsilon,
                        constants,
                        functions,
                        aliases,
                        line_num,
                        var_names,
                        do_simplify,
                    )

            if op_lower in ["+", "*"] and len(simplified_operands) > 2:
                result = simplified_operands[-1]
                for operand in reversed(simplified_operands[:-1]):
                    result = [op_lower, operand, result]
                return result

            return [op_lower] + simplified_operands

        if op_lower in OPERATION_EXPANSIONS:
            expansion = OPERATION_EXPANSIONS[op_lower]
            operands = ast[1:]
            simplified_operands = [
                simplify_ast(
                    node,
                    epsilon,
                    constants,
                    functions,
                    aliases,
                    line_num,
                    var_names,
                    do_simplify,
                )
                for node in operands
            ]

            if op_lower in ["!", "frac", "nat", "max0", "min0"]:
                expanded = expansion(simplified_operands[0])
            elif op_lower in ["=0", "sign"]:
                expanded = expansion(simplified_operands[0], epsilon)
            elif op_lower in ["==", ">", ">=", "<", "<="]:
                expanded = expansion(
                    simplified_operands[0], simplified_operands[1], epsilon
                )
            elif op_lower in ["max", "min", "mod"]:
                expanded = expansion(simplified_operands[0], simplified_operands[1])
            elif op_lower == "if":
                expanded = expansion(
                    simplified_operands[0],
                    simplified_operands[1],
                    simplified_operands[2],
                )
            else:
                raise ParserError(f"unknown expansion for '{op_lower}'", line_num)

            print(
                simplify_ast(
                    expanded,
                    epsilon,
                    constants,
                    functions,
                    aliases,
                    line_num,
                    var_names,
                    do_simplify,
                )
            )

            return simplify_ast(
                expanded,
                epsilon,
                constants,
                functions,
                aliases,
                line_num,
                var_names,
                do_simplify,
            )

        raise ParserError(f"unknown operation '{op_lower}'", line_num)

    raise ParserError(f"invalid operator: {op}", line_num)


@cached("sympy")
def simplify_with_sympy(expr_str: str, line_num: int) -> str:
    try:
        import sympy as sp

        expr_str = (
            expr_str.replace("arcsin", "asin")
            .replace("arccos", "acos")
            .replace("arctan", "atan")
        )

        sympy_expr = sp.sympify(expr_str)
        simplified = sp.simplify(sympy_expr)
        result = str(simplified)

        result = (
            result.replace("asin", "arcsin")
            .replace("acos", "arccos")
            .replace("atan", "arctan")
        )

        debug_print(f"sympy simplified: {len(expr_str)} -> {len(result)} chars")
        return result.replace(" ", "")
    except Exception as e:
        debug_print(f"sympy failed, keeping original: {e}")
        return expr_str


@cached("length")
def expr_length(ast: Any, line_num: int) -> int:
    if isinstance(ast, (int, float)):
        return len(str(ast))
    if isinstance(ast, str):
        return len(ast)
    if isinstance(ast, list) and len(ast) == 1:
        return expr_length(ast[0], line_num)
    if not isinstance(ast, list) or len(ast) < 2:
        raise ParserError(f"invalid ast node: {ast}", line_num)

    op = ast[0]
    operands = ast[1:]

    if not isinstance(op, str):
        raise ParserError(f"operator must be string, got {type(op)}", line_num)

    if op in ["+", "-", "*", "/", "**"]:
        if len(operands) != 2:
            raise ParserError(f"operator '{op}' requires exactly 2 operands", line_num)
        left_len = expr_length(operands[0], line_num)
        right_len = expr_length(operands[1], line_num)

        needs_right_parens = (
            isinstance(operands[1], list)
            and len(operands[1]) >= 2
            and operands[1][0] in ["+", "-", "*", "/", "**"]
            and operands[1][0] != op
        )
        needs_left_parens = (
            op in ["-", "/", "**"]
            and isinstance(operands[0], list)
            and len(operands[0]) >= 2
            and operands[0][0] in ["+", "-", "*", "/", "**"]
        )

        total = left_len + len(op) + right_len
        if needs_left_parens:
            total += 2
        if needs_right_parens:
            total += 2
        return total

    if op in BUILTIN_FUNCS:
        if len(operands) != 1:
            raise ParserError(f"function '{op}' requires exactly 1 operand", line_num)
        return len(op) + 1 + expr_length(operands[0], line_num) + 1

    raise ParserError(f"unknown basic operation '{op}' in pass 2", line_num)


@cached("generate")
def generate_expression(ast: Any, line_num: int) -> str:
    if isinstance(ast, (int, float)):
        return str(ast)
    if isinstance(ast, str):
        return ast
    if isinstance(ast, list) and len(ast) == 1:
        return generate_expression(ast[0], line_num)
    if not isinstance(ast, list) or len(ast) < 2:
        raise ParserError(f"invalid ast node: {ast}", line_num)

    op = ast[0]
    operands = ast[1:]

    if not isinstance(op, str):
        raise ParserError(f"operator must be string, got {type(op)}", line_num)

    if op in ["+", "-", "*", "/", "**"]:
        if len(operands) != 2:
            raise ParserError(f"operator '{op}' requires exactly 2 operands", line_num)

        left = generate_expression(operands[0], line_num)
        right = generate_expression(operands[1], line_num)

        needs_right_parens = (
            isinstance(operands[1], list)
            and len(operands[1]) >= 2
            and operands[1][0] in OP_PRECEDENCE
            and OP_PRECEDENCE[operands[1][0]] < OP_PRECEDENCE[op]
        )
        needs_left_parens = (
            isinstance(operands[0], list)
            and len(operands[0]) >= 2
            and operands[0][0] in OP_PRECEDENCE
            and (
                OP_PRECEDENCE[operands[0][0]] < OP_PRECEDENCE[op]
                or (
                    op in ["-", "/", "**"]
                    and OP_PRECEDENCE[operands[0][0]] <= OP_PRECEDENCE[op]
                )
            )
        )

        if needs_left_parens:
            left = f"({left})"
        if needs_right_parens:
            right = f"({right})"

        return f"{left}{op}{right}"

    if op in BUILTIN_FUNCS:
        if len(operands) != 1:
            raise ParserError(f"function '{op}' requires exactly 1 operand", line_num)
        inner = generate_expression(operands[0], line_num)
        return f"{op}({inner})"

    raise ParserError(f"unknown basic operation '{op}' in pass 2", line_num)


@cached("try_simplify")
def try_simplify(ast, inst):
    if not isinstance(ast, list):
        return (ast, isinstance(ast, str))

    if len(ast) == 1:
        return try_simplify(ast[0], inst)

    op = ast[0]
    args = ast[1:]

    simplified_args = []
    has_any_var = False

    for arg in args:
        simp_arg, has_var = try_simplify(arg, inst)
        simplified_args.append(simp_arg)
        if has_var:
            has_any_var = True

    if has_any_var:
        return ([op] + simplified_args, True)

    expr = generate_expression([op] + simplified_args, -1)

    try:
        result_val = round(eval(expr), inst.precision)
        if result_val.is_integer():
            result_val = int(result_val)
        debug_print(f"simplified literal expression to {result_val}")
        return (result_val, False)
    except Exception as e:
        debug_print(f"failed to simplify literal: {e}")
        return ([op] + simplified_args, True)


def parse_pragma(
    line: str, config: ProgramConfig, line_num: int, current_phase: ParsePhase
) -> Union[Tuple[bool, ParsePhase], Tuple[bool, ParsePhase, tuple]]:
    if not line.startswith("//"):
        return False, current_phase

    first_split = line.split(None, 1)
    if len(first_split) < 1:
        return False, current_phase

    pragma = first_split[0][2:].lower()
    rest_of_line = first_split[1] if len(first_split) > 1 else ""

    if pragma == "precision":
        if not rest_of_line:
            raise ParserError("precision requires a value", line_num)
        config.precision = int(rest_of_line.strip())
        config.current_precision = config.precision

    elif pragma == "epsilon":
        if not rest_of_line:
            raise ParserError("epsilon requires a value", line_num)
        config.epsilon = rest_of_line.strip()
        config.current_epsilon = config.epsilon

    elif pragma == "simplify":
        if not rest_of_line:
            raise ParserError(f"{pragma} requires true/false", line_num)
        value = rest_of_line.strip().lower() == "true"
        config.simplify_literals = value
        config.current_simplify = value

    elif pragma == "dupe":
        if not rest_of_line:
            raise ParserError(f"{pragma} requires true/false", line_num)
        parts = rest_of_line.strip().split()
        value = parts[0].lower() == "true"
        config.detect_duplicates = value
        config.current_dupe = value
        if len(parts) >= 2:
            try:
                min_savings = int(parts[1])
                config.dupe_min_savings = min_savings
                config.current_dupe_min_savings = min_savings
            except ValueError:
                raise ParserError(f"dupe min_savings must be an integer", line_num)

    elif pragma == "sympy":
        if not rest_of_line:
            raise ParserError(f"{pragma} requires true/false", line_num)
        value = rest_of_line.strip().lower() == "true"
        config.use_sympy = value
        config.current_sympy = value

    elif pragma == "variables":
        if not rest_of_line:
            raise ParserError("variables requires variable names", line_num)
        var_list = rest_of_line.strip().split()
        if "ans" in var_list:
            raise ParserError("cannot override 'ans' in //variables", line_num)
        config.var_names = var_list + ["ans", "pi", "e"]

    elif pragma == "alias":
        parts = rest_of_line.split(None, 1)
        if len(parts) < 2:
            raise ParserError("alias requires alias_name and target_var", line_num)
        alias_name = parts[0].lower()
        target_var = parts[1].lower()
        if not alias_name.replace("_", "").isalnum():
            raise ParserError(f"invalid alias name '{alias_name}'", line_num)
        if alias_name in config.aliases:
            return True, current_phase
        if target_var not in config.var_names:
            raise ParserError(
                f"alias target '{target_var}' must be a valid variable", line_num
            )
        config.aliases[alias_name] = target_var

    elif pragma == "global":
        parts = rest_of_line.split(None, 1)
        if len(parts) < 2:
            raise ParserError("global requires name and value", line_num)
        const_name = parts[0]
        const_value_str = parts[1]
        config.constants[const_name] = compile_value(
            const_value_str, config.constants, line_num, config
        )

    elif pragma == "display":
        config.output_mode = ("display", None)

    elif pragma == "store":
        if not rest_of_line:
            raise ParserError("store requires a variable name", line_num)
        used_variable = rest_of_line.split()[0].lower()
        config.output_mode = ("store", config.aliases.get(used_variable, used_variable))

    elif pragma == "ret":
        if not rest_of_line:
            raise ParserError("ret requires an expression", line_num)
        return True, current_phase, ("ret", rest_of_line)

    elif pragma == "repeat":
        parts = rest_of_line.split(None, 1)
        if len(parts) < 1:
            raise ParserError(
                "repeat requires range value and optional iterator name", line_num
            )
        range_expr = parts[0]
        var_name = parts[1].split()[0] if len(parts) >= 2 else None
        try:
            range_val = int(
                compile_value(range_expr, config.constants, line_num, config)
            )
            if range_val < 0:
                raise ParserError(
                    f"repeat range must be non-negative, got {range_val}", line_num
                )
        except Exception as e:
            raise ParserError(f"failed to evaluate repeat range: {e}", line_num)
        if var_name and not var_name.replace("_", "").isalnum():
            raise ParserError(f"invalid iterator name '{var_name}'", line_num)
        loop_ctx = LoopContext(
            range_val=range_val, var_name=var_name, start_line=line_num
        )
        config.loop_stack.append(loop_ctx)
        return True, current_phase, ("repeat_start",)

    elif pragma == "endrepeat":
        if not config.loop_stack:
            raise ParserError("//endrepeat without matching //repeat", line_num)
        return True, current_phase, ("repeat_end",)

    elif pragma == "local":
        if not config.loop_stack:
            raise ParserError(
                "//local can only be used inside //repeat loops", line_num
            )
        parts = rest_of_line.split(None, 1)
        if len(parts) < 2:
            raise ParserError("local requires name and value", line_num)
        local_name = parts[0]
        local_expr = parts[1]
        if not local_name.replace("_", "").isalnum():
            raise ParserError(f"invalid local name '{local_name}'", line_num)
        check_naming_conflicts(local_name, config, line_num, "local ")
        current_loop = config.loop_stack[-1]
        current_loop.locals_in_scope.add(local_name)
        return True, current_phase, ("local_def", local_name, local_expr)

    return True, current_phase


def collect_loop_body(
    code: List[str], start_idx: int
) -> Tuple[List[Tuple[int, str]], int]:
    loop_lines = []
    i = start_idx
    depth = 0

    verbose_print(f"collecting loop body starting at line {start_idx + 1}")

    while i < len(code):
        line = code[i].split("#")[0].strip()
        line_num = i + 1
        if not line:
            i += 1
            continue

        if line.lower().startswith("//repeat"):
            depth += 1
            debug_print(f"nested repeat, depth now {depth + 1}")
        elif line.lower().startswith("//endrepeat"):
            if depth == 1:
                loop_lines.append((line_num, line))
                verbose_print(f"collected loop body: {len(loop_lines)} lines")
                return loop_lines, i - start_idx + 1
            depth -= 1
            debug_print(f"nested endrepeat, depth now {depth + 1}")

        loop_lines.append((line_num, line))
        i += 1

    raise ParserError("//repeat without matching //endrepeat", start_idx + 1)


def unwrap_loop(
    loop_lines: List[Tuple[int, str]], config: ProgramConfig, start_line: int
) -> List[Tuple[int, str]]:
    if not loop_lines:
        return []

    first_line_num, first_line = loop_lines[0]
    parts = first_line.split(None, 3)

    if len(parts) < 2:
        raise ParserError("repeat requires range value", first_line_num)

    range_expr = parts[1]
    iter_var = parts[2] if len(parts) >= 3 else None

    try:
        range_val = int(
            compile_value(range_expr, config.constants, first_line_num, config)
        )
    except Exception as e:
        raise ParserError(f"failed to evaluate repeat range: {e}", first_line_num)

    verbose_print(f"unwrapping loop: {range_val} iterations")

    body_lines = loop_lines[1:-1]
    local_defs = []
    non_local_lines = []

    for line_num, line in body_lines:
        if line.lower().startswith("//local"):
            rest = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
            parts = rest.split(None, 1)
            if len(parts) < 2:
                raise ParserError("local requires name and value", line_num)
            local_name = parts[0]
            local_expr = parts[1]
            local_defs.append((local_name, local_expr, line_num))
            debug_print(f"found local definition: {local_name}")
        else:
            non_local_lines.append((line_num, line))

    unwrapped = []

    for iteration in range(range_val):
        debug_print(f"processing iteration {iteration}/{range_val}")
        iteration_constants = dict(config.constants)
        loop_vars = {}

        if iter_var:
            iteration_constants[iter_var] = iteration
            loop_vars[iter_var] = iteration

        for local_name, local_expr, def_line_num in local_defs:
            try:
                iteration_constants[local_name] = compile_value(
                    local_expr, iteration_constants, def_line_num, config, loop_vars
                )
                loop_vars[local_name] = iteration_constants[local_name]
            except Exception as e:
                raise ParserError(
                    f"failed to evaluate local '{local_name}': {e}", def_line_num
                )

        for line_num, line in non_local_lines:
            new_line = line
            for const_name, const_val in iteration_constants.items():
                new_line = re.sub(
                    r"\b" + re.escape(const_name) + r"\b", str(const_val), new_line
                )

            unwrapped.append((line_num, new_line))

    verbose_print(
        f"unwrapped loop into {len(unwrapped)} lines ({range_val} iterations)"
    )
    return unwrapped


def replace_in_ast(ast: Any, pattern: Any, replacement: str) -> Any:
    if ast == pattern:
        return replacement
    if isinstance(ast, list):
        return [replace_in_ast(x, pattern, replacement) for x in ast]
    return ast


def find_beneficial_duplicates(
    ast: Any, min_length: int = 4, min_savings: int = -999
) -> List[Tuple[int, int, Any]]:
    seen = {}
    for sub in all_sublists(ast):
        key = json.dumps(sub, sort_keys=False)
        seen.setdefault(key, {"count": 0, "obj": sub})
        seen[key]["count"] += 1

    beneficial_dupes = []
    for v in seen.values():
        if v["count"] > 1:
            dupe_length = expr_length(v["obj"], -1)
            if dupe_length >= min_length:
                total_size = v["count"] * dupe_length
                savings = total_size - (dupe_length + 3 + (v["count"] - 1) * 3)
                if savings > min_savings:
                    beneficial_dupes.append((savings, total_size, v["obj"]))

    beneficial_dupes.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return beneficial_dupes


def extract_subexpressions(
    ast: Any, initial_length: int, min_savings: int = -999
) -> List[Tuple[Any, Optional[str]]]:
    current_ast = ast

    beneficial_dupes = find_beneficial_duplicates(current_ast, min_savings=min_savings)

    if not beneficial_dupes:
        return [(current_ast, None)]

    largest_dupe = beneficial_dupes[0][2]
    replaced_ast = replace_in_ast(current_ast, largest_dupe, "ans")

    debug_print(
        f"extracted single best duplicate with savings: {beneficial_dupes[0][0]}"
    )
    return [(largest_dupe, "ans"), (replaced_ast, None)]


def preprocess_globals_and_aliases(code: List[str], config: ProgramConfig):
    verbose_print("preprocessing globals and aliases")

    for i, line in enumerate(code):
        line = line.split("#")[0].strip()
        if line and line.lower().startswith("//variables"):
            parse_pragma(line, config, i + 1, ParsePhase.constants)

    for i, line in enumerate(code):
        line = line.split("#")[0].strip()
        if line and (
            line.lower().startswith("//global") or line.lower().startswith("//alias")
        ):
            parse_pragma(line, config, i + 1, ParsePhase.constants)

    debug_print(f"found {len(config.constants)} globals, {len(config.aliases)} aliases")


def expand_loops(code: List[str], config: ProgramConfig) -> List[Tuple[int, str]]:
    verbose_print("expanding loops")

    expanded_code = []
    i = 0

    while i < len(code):
        line = code[i].split("#")[0].strip()
        if not line:
            i += 1
            continue

        if line.lower().startswith("//repeat"):
            loop_lines, lines_consumed = collect_loop_body(code, i)
            unwrapped = unwrap_loop(loop_lines, config, i + 1)
            expanded_code.extend(unwrapped)
            i += lines_consumed
        else:
            expanded_code.append((i + 1, line))
            i += 1

    verbose_print(f"expanded to {len(expanded_code)} lines")
    return expanded_code


def process_instructions(
    expanded_code: List[Tuple[int, str]], config: ProgramConfig
) -> List[Instruction]:
    instructions = []
    i = 0

    while i < len(expanded_code):
        line_num, line = expanded_code[i]

        if line.lower().startswith("//") and not line.lower().startswith("//ret"):
            pragma_result = parse_pragma(line, config, line_num, ParsePhase.config)
            if isinstance(pragma_result, tuple) and len(pragma_result) == 3:
                if pragma_result[2][0] in ["repeat_start", "repeat_end", "local_def"]:
                    i += 1
                    continue
            i += 1
            continue

        if config.output_mode is None:
            raise ParserError(
                "no output mode before codeblock (use //display or //store)", line_num
            )

        block, lines_consumed = collect_codeblock(expanded_code, i, config)
        final_expr = expand_intermediates(
            block.ret_expr, block.intermediates, block.line_end
        )

        try:
            tokens = tokenize(final_expr)
            ast = parse_tokens(tokens, block.line_start)
        except ParserError as e:
            raise ParserError(str(e), block.line_start)

        instructions.append(
            Instruction(
                ast=ast,
                output=config.output_mode,
                epsilon=config.current_epsilon,
                precision=config.current_precision,
                simplify=config.current_simplify,
                sympy=config.current_sympy,
                dupe=config.current_dupe,
                dupe_min_savings=config.current_dupe_min_savings,
                line_start=block.line_start,
                line_end=block.line_end,
            )
        )

        config.output_mode = None
        i += lines_consumed

    return instructions


def compile_instructions(
    instructions: List[Instruction], config: ProgramConfig
) -> List[Tuple[str, Tuple[str, Optional[str]]]]:
    results = []

    for inst_idx, inst in enumerate(instructions):
        try:
            simplified_ast = simplify_ast(
                inst.ast,
                "ε",
                config.constants,
                config.functions,
                config.aliases,
                inst.line_start,
                config.var_names,
                inst.simplify,
            )
            initial_length = expr_length(simplified_ast, -1)

            subexpressions = (
                extract_subexpressions(
                    simplified_ast, initial_length, inst.dupe_min_savings
                )
                if inst.dupe
                else [(simplified_ast, None)]
            )

            for index, (expr_ast, var_name) in enumerate(subexpressions):
                if inst.simplify:
                    expr_ast = try_simplify(expr_ast, inst)[0]

                expr = generate_expression(expr_ast, inst.line_start)

                if expr.startswith("(") and expr.endswith(")"):
                    paren_depth = 0
                    can_strip = True
                    for char in expr[1:-1]:
                        if char == "(":
                            paren_depth += 1
                        elif char == ")":
                            paren_depth -= 1
                            if paren_depth < 0:
                                can_strip = False
                                break
                    if can_strip:
                        expr = expr[1:-1]

                epsilon_str = str(inst.epsilon)
                try:
                    float(epsilon_str)
                    needs_parens = False
                except ValueError:
                    needs_parens = any(op in epsilon_str for op in ["+", "-", "*", "/"])

                if needs_parens:
                    epsilon_str = f"({epsilon_str})"
                expr = expr.replace("ε", epsilon_str)

                if inst.sympy:
                    expr = simplify_with_sympy(expr, inst.line_start)

                if var_name == "ans":
                    output_mode = ("store", "ans", "dupe")
                else:
                    output_mode = inst.output

                results.append((expr, output_mode))

        except ParserError as e:
            if e.line_num is None:
                raise ParserError(str(e), inst.line_start)
            raise

    return results


def parse_program(code: List[str]) -> List[Tuple[str, Tuple[str, Optional[str]]]]:
    verbose_print("starting program parsing")

    config = ProgramConfig()

    preprocess_globals_and_aliases(code, config)
    expanded_code = expand_loops(code, config)
    instructions = process_instructions(expanded_code, config)
    results = compile_instructions(instructions, config)

    verbose_print("parsing complete!")
    return results


def main():
    import sys

    if LOGFILE:
        with open("logfile.txt", "w") as f:
            f.seek(0)

    filepath = sys.argv[1] if len(sys.argv) >= 2 else "testfile.pm"

    try:
        print(f"=== Paramath Compiler v2.0.0 ===")
        print(f"reading: {filepath}\n")

        with open(filepath) as f:
            code = f.read().strip().replace(";", "\n").split("\n")

        results = parse_program(code)

        with open("math.txt", "w") as f:
            for result, output in results:
                result = (
                    result.replace("**", "^").replace("*", "").replace("ans", "ANS")
                )
                print(f"to {output}:")
                print(result)
                f.write(f"to {output}:\n{result}\n")

        print(f"\n=== compilation successful! ===")
        print(f"generated {len(results)} expressions")

    except FileNotFoundError:
        print(f"error: file '{filepath}' not found")
        sys.exit(1)
    except ParserError as e:
        print(f"parser error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
