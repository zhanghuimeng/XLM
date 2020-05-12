import argparse

parser = argparse.ArgumentParser(description='Convert output to upload format')
parser.add_argument("--test_output", type=str, default="",
                    help="Test output file name")
parser.add_argument("--name", type=str, default="",
                    help="method name")
parser.add_argument("--pair", type=str, default="",
                    help="language pair")
parser.add_argument("--output", type=str, default="",
                    help="Output file name")

args = parser.parse_args()

with open(args.test_output, "r") as f:
    lines = f.readlines()

# line number starts from zero
with open(args.output, "w") as f:
    for i, line in enumerate(lines):
        r = line.strip()
        f.write("%s\t%s\t%d\t%s\n" % (args.pair, args.name, i, r))
