import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import paramath
from paramath import ParserError


def test_compile_all_pm_files():
    paramath.ARGS = {
        "verbose": False,
        "debug": False,
        "logfile": None,
    }
    test_dir = Path(__file__).parent.parent / "paramath_tests"

    if not test_dir.exists():
        print(f"uh oh, can't find test directory: {test_dir}")
        return False

    pm_files = list(test_dir.rglob("*.pm"))

    if not pm_files:
        print(f"no .pm files found in {test_dir} dkajsdljad")
        return False

    print(f"found {len(pm_files)} .pm files to test :3")
    print()

    failed_files = []
    passed_files = []

    for pm_file in sorted(pm_files):
        relative_path = pm_file.relative_to(test_dir)
        print(f"testing {relative_path}... ", end="", flush=True)

        try:

            with open(pm_file, "r") as f:
                code = f.read().strip().replace(";", "\n").split("\n")

            results = paramath.parse_program(code, safe_eval=False)

            print(f"✓ ({len(results)} expressions)")
            passed_files.append(relative_path)

        except ParserError as e:
            print(f"✗ parser error: {e}")
            failed_files.append((relative_path, str(e)))

        except Exception as e:
            print(f"✗ unexpected error: {e}")
            failed_files.append((relative_path, f"unexpected: {e}"))

    print()
    print("=" * 60)
    print(f"results: {len(passed_files)}/{len(pm_files)} passed")

    if failed_files:
        print()
        print("failed files:")
        for file, error in failed_files:
            print(f"  - {file}")
            print(f"    {error}")
        return False
    else:
        print()
        print("all tests passed!! GHRTJHGRJHRJ >:3c")
        return True


def main():
    success = test_compile_all_pm_files()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
