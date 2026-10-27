"""YANDF output blocks (write_block.f90).

Each function writes '#'-prefixed key/value lines to an open text file, with
the same indentation and number formats as the Fortran.
"""
from __future__ import annotations

from typing import TextIO

from .fortran import es


def _prefix(indent: int) -> str:
    return "# " + " " * indent


def write_char(fh: TextIO, indent: int, key: str, word: str) -> None:
    fh.write(f"{_prefix(indent)}{key.rstrip()}: {word.rstrip()}\n")


def write_real(fh: TextIO, indent: int, key: str, x) -> None:
    fh.write(f"{_prefix(indent)}{key.rstrip()}: {es(x, 13, 6)}\n")


write_double = write_real


def write_integer(fh: TextIO, indent: int, key: str, k: int) -> None:
    fh.write(f"{_prefix(indent)}{key.rstrip()}: {int(k)}\n")


def write_header(fh: TextIO, indent: int, title: str, source: str, user: str, date: str,
                 comment: str) -> None:
    write_char(fh, indent, "header", "")
    id2 = indent + 2
    write_char(fh, id2, "title", title.rstrip())
    write_char(fh, id2, "source", source.rstrip())
    if user.strip():
        write_char(fh, id2, "user", user.rstrip())
    write_char(fh, id2, "date", date.rstrip())
    if comment.strip():
        write_char(fh, id2, "format", comment.rstrip())


def write_nuc(fh: TextIO, indent: int, Z: int, A: int, nuclide: str) -> None:
    if Z > 0:
        write_integer(fh, indent, "Z", Z)
    write_integer(fh, indent, "A", A)
    if Z > 0:
        write_char(fh, indent, "nuclide", nuclide.rstrip())


def write_target(fh: TextIO, indent: int, Ztarget: int, Atarget: int, targetnuclide: str) -> None:
    write_char(fh, indent, "target", "")
    write_nuc(fh, indent + 2, Ztarget, Atarget, targetnuclide)


def write_reaction(fh: TextIO, indent: int, reaction: str, Q: float, E: float,
                   MF: int, MT: int) -> None:
    id2 = indent + 2
    write_char(fh, indent, "reaction", "")
    write_char(fh, id2, "type", reaction.rstrip())
    if Q != 0.0 or E != 0.0:
        write_double(fh, id2, "Q-value [MeV]", Q)
        write_double(fh, id2, "E-threshold [MeV]", E)
    if MF > 0 and MT > 0:
        write_integer(fh, id2, "ENDF_MF", MF)
        write_integer(fh, id2, "ENDF_MT", MT)


def write_residual(fh: TextIO, indent: int, Z: int, A: int, nuclide: str) -> None:
    write_char(fh, indent, "residual", "")
    write_nuc(fh, indent + 2, Z, A, nuclide)


def write_quantity(fh: TextIO, indent: int, quantity: str) -> None:
    write_char(fh, indent, "quantity", "")
    write_char(fh, indent + 2, "type", quantity.rstrip())


def _centered(text: str, width: int, brackets: bool) -> str:
    """One 15-character column header, built like write_datablock."""
    L = len(text.rstrip())
    word = [" "] * width
    ibeg = max(width // 2 - L // 2, 0)
    word[ibeg:ibeg + L] = list(text[:L])
    if brackets:
        if ibeg >= 1:
            word[ibeg - 1] = "["
        if ibeg + L < width:
            word[ibeg + L] = "]"
    return "".join(word[:width])


def write_datablock(fh: TextIO, indent: int, Nc: int, Ne: int, col: list[str],
                    un: list[str]) -> None:
    write_char(fh, indent, "datablock", "")
    id2 = indent + 2
    write_integer(fh, id2, "columns", Nc)
    write_integer(fh, id2, "entries", Ne)
    width = 15
    obs = "##" + "".join(_centered(col[k], width, False) for k in range(Nc))
    units = "##" + "".join(_centered(un[k], width, True) for k in range(Nc))
    fh.write(obs.rstrip() + "\n")
    fh.write(units.rstrip() + "\n")
