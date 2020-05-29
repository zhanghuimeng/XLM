import argparse

parser = argparse.ArgumentParser(description='Mark wordpieces, start=1, others=0')
parser.add_argument("--input", type=str, default="",
                    help="Sentences after BPE")
parser.add_argument("--output", type=str, default="",
                    help="Marked word piece")

args = parser.parse_args()

with open(args.input, "r") as f:
    sentences = [line.rstrip() for line in f]

marks = []
for sentence in sentences:
    tokens = sentence.split(" ")
    marks.append([])
    for i, token in enumerate(tokens):
        if i == 0 or not tokens[i-1].endswith("@@"):
            marks[-1].append("1")
        else:
            marks[-1].append("0")

with open(args.output, "w") as f:
    for mark in marks:
        f.write("%s\n" % " ".join(mark))
