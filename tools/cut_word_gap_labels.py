import argparse

parser = argparse.ArgumentParser(description='Cut or convert labels')
parser.add_argument("--input", type=str, default="",
                    help="Input file name")
parser.add_argument("--word_output", type=str, default="",
                    help="Word tags output")
parser.add_argument("--gap_output", type=str, default=None,
                    help="Gap tags output")
parser.add_argument('--cut', action='store_true',
                    help="Whether to cut the file into word and gap tags or not")

args = parser.parse_args()

if args.cut:
    if not args.gap_output:
        raise ValueError("Did not give two filenames")

lines = []
with open(args.input, "r") as f:
    for l in f:
        tokens = l.rstrip().split(" ")
        lines.append([])
        for token in tokens:
            if token == "OK":
                lines[-1].append(1)
            else:
                lines[-1].append(0)

if not args.cut:
    with open(args.word_output, "w") as f:
        for line in lines:
            f.write("%s\n" % " ".join(map(str, line)))
else:
    # word tags
    with open(args.word_output, "w") as f:
        for line in lines:
            f.write("%s\n" % " ".join(map(str, line[1::2])))
    # gap tags
    with open(args.gap_output, "w") as f:
        for line in lines:
            f.write("%s\n" % " ".join(map(str, line[0::2])))
