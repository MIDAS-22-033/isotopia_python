"""Example: inspect an ISOTOPIA input file and an isotope .act file."""
from pathlib import Path
import argparse
from isotopia_py import read_act, read_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to isotopia.inp")
    parser.add_argument("--act", required=True, help="path to an isotope .act file")
    args = parser.parse_args()

    if args.input:
        print("Input settings:")
        for key, value in read_input(args.input).items():
            print(f"  {key}: {value}")

    act = read_act(args.act)
    print(f"\nFile: {act.path}")
    print(f"Data points: {len(act)}")
    print("Selected metadata:")
    for key in (
        "title", "E-Beam [MeV]", "E-Back [MeV]", "Beam current [mA]",
        "Number of target atoms", "Reaction constant [s^-1]",
        "Decay constant [s^-1]", "Irradiation time [s]", "Cooling time [s]",
    ):
        if key in act.metadata:
            print(f"  {key}: {act.metadata[key]}")

    if len(act):
        last = len(act) - 1
        print("\nFirst data point:")
        print(f"  time = {act.column('time_s')[0]:.6g} s")
        print(f"  activity = {act.column('activity_bq')[0]:.6g} Bq")
        print(f"  inventory = {act.column('isotope_inventory')[0]:.6g} atoms")
        print("\nLast data point:")
        print(f"  time = {act.column('time_s')[last]:.6g} s")
        print(f"  activity = {act.column('activity_bq')[last]:.6g} Bq")
        print(f"  inventory = {act.column('isotope_inventory')[last]:.6g} atoms")


if __name__ == "__main__":
    main()
