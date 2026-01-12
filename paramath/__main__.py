from . import PROGRAM_VERSION, parse_program, ParserError, ARGS
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Paramath Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  paramath testfile.pm
  paramath testfile.pm -d -V
  paramath testfile.pm -l output.log
  paramath testfile.pm -dVl debug.log
        """,
    )

    parser.add_argument(
        "filepath",
        nargs="?",
        help="Input paramath file",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Paramath {PROGRAM_VERSION}",
        help="prints the Paramath version number and exits",
    )
    parser.add_argument(
        "-O",
        "--output",
        required=False,
        metavar="FILE",
        help="output to a file INSTEAD of printing",
    )
    parser.add_argument(
        "-P",
        "--print",
        action="store_true",
        help="print success messages (always on when --output is set)",
    )
    parser.add_argument(
        "-D", "--debug", action="store_true", help="enable debug output"
    )
    parser.add_argument(
        "-V", "--verbose", action="store_true", help="enable verbose output"
    )
    parser.add_argument(
        "-S",
        "--safe-eval",
        action="store_true",
        help="prints and blocks python code from evaluating and exits, used for safely running unknown scripts",
    )
    parser.add_argument(
        "-L", "--logfile", required=False, metavar="FILE", help="write logs to FILE"
    )
    args = parser.parse_args()
    for name in ARGS:
        if hasattr(args, name):
            ARGS[name] = getattr(args, name)

    if ARGS["logfile"]:
        with open(ARGS["logfile"], "w") as f:
            f.write(f"Paramath Compiler {PROGRAM_VERSION}\n")

    try:
        if args.filepath is None:
            raise ParserError("No path to file provided, quitting")
        if args.print or args.output:
            print(f"reading {args.filepath}")
            if args.safe_eval:
                print("[safe evaluation enabled]")
            if ARGS["debug"]:
                print("[debug mode enabled]")
            if ARGS["verbose"]:
                print("[verbose mode enabled]")
            if ARGS["logfile"]:
                print(f"[logging to: {ARGS['logfile']}]")

        with open(args.filepath) as f:
            code = f.read().strip().replace(";", "\n").split("\n")

        results = parse_program(code, args.safe_eval)

        if args.print or args.output:
            print("=== compilation successful! ===")
            print(f"generated {len(results)} expressions")

        out = ""
        for result, output in results:
            result = (
                result.replace("**", "^").replace("*", "").replace("ans", "ANS")
            )
            out += f"to {output}:\n{result}\n"

        if args.output:
            with open(args.output, "w+") as f:
                f.write(out)
            if args.print:
                print(f"written to: {args.output}")
        else:
            if args.print:
                print()
            print(out, end="", flush=True)

    except FileNotFoundError:
        print(f"error: file '{args.filepath}' not found")
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
