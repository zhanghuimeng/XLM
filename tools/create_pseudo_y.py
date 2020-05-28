import argparse
import os

parser = argparse.ArgumentParser(description='Create pesudo y values')
parser.add_argument("--s1", type=str, default="",
                    help="Sentence 1")
parser.add_argument("--s2", type=str, default="",
                    help="Sentence 2")
parser.add_argument("--type", type=str, default="",
                    help="Type of pesudo data")
parser.add_argument("--name", type=str, default="",
                    help="Output name")
parser.add_argument("--output", type=str, default="",
                    help="Output dir")

args = parser.parse_args()

with open(args.s1, "r") as f:
    s1 = [line.rstrip() for line in f]
with open(args.s2, "r") as f:
    s2 = [line.rstrip() for line in f]

if args.type == "DA" or args.type == "HTER":
    l = len(s1)
    with open(os.path.join(args.output, "%s.test.label" % args.name), "w") as f:
        for i in range(l):
            f.write("%f\n" % 0.5)
elif args.type == "TAG":
    l = len(s1)
    with open(os.path.join(args.output, "%s.test.src_tags" % args.name), "w") as f:
        for i in range(l):
            a = ["1"] * len(s1[i].split())
            f.write("%s\n" % " ".join(a))

    l = len(s2)
    with open(os.path.join(args.output, "%s.test.tgt_tags" % args.name), "w") as f:
        for i in range(l):
            a = ["1"] * len(s2[i].split())
            f.write("%s\n" % " ".join(a))

    l = len(s2)
    with open(os.path.join(args.output, "%s.test.gap_tags" % args.name), "w") as f:
        for i in range(l):
            a = ["1"] * (len(s2[i].split()) + 1)
            f.write("%s\n" % " ".join(a))
else:
    raise ValueError("No Such Type")
