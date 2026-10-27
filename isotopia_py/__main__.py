"""Command line: ``python -m isotopia_py [input] [-o outdir]``.

Like the Fortran ``isotopia < isotopia.inp > isotopia.out``, the input is read
from standard input when no file is given and the summary goes to standard
output; .act and other files are written to the output directory.
"""
from __future__ import annotations

import argparse
import sys

from .run import run
from .state import IsotopiaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="isotopia_py", description=__doc__)
    parser.add_argument("input", nargs="?", help="input file (default: standard input)")
    parser.add_argument("-o", "--outdir", default=".", help="directory for output files")
    args = parser.parse_args(argv)
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    try:
        run(lines, outdir=args.outdir)
    except IsotopiaError as exc:
        sys.stdout.flush()
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
