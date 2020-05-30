import argparse
import os

parser = argparse.ArgumentParser(description='Convert output to upload format')
parser.add_argument("--source_sentences", type=str, default="",
                    help="Original source sentence")
parser.add_argument("--target_sentences", type=str, default="",
                    help="Original target sentence")
parser.add_argument("--test_src_output", type=str, default="",
                    help="Test src output")
parser.add_argument("--test_tgt_output", type=str, default="",
                    help="Test tgt output")
parser.add_argument("--test_gap_output", type=str, default="",
                    help="Test gap output")
parser.add_argument("--pair", type=str, default="",
                    help="language pair")
parser.add_argument("--name", type=str, default="",
                    help="method pair")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output file dir")

args = parser.parse_args()

with open(args.source_sentences, "r") as f:
    source_len = []
    source = []
    for line in f:
        source.append(line.rstrip().split(" "))
        source_len.append(len(source[-1]))
with open(args.target_sentences, "r") as f:
    target_len = []
    target = []
    for line in f:
        target.append(line.rstrip().split(" "))
        target_len.append(len(target[-1]))

with open(args.test_src_output, "r") as f:
    src_tags = []
    for line in f:
        src_tags += line.rstrip().split(" ")
with open(os.path.join(args.output_dir, "predictions_src.txt"), "w") as f:
    i = 0
    for j, slen in enumerate(source_len):
        for k in range(slen):
            f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\n" % (args.pair, args.name, "src", j, k, source[j][k], src_tags[i]))
            i += 1

with open(args.test_tgt_output, "r") as f:
    tgt_tags = []
    for line in f:
        tgt_tags += line.rstrip().split(" ")
with open(os.path.join(args.output_dir, "predictions_mt.txt"), "w") as f:
    i = 0
    for j, slen in enumerate(target_len):
        for k in range(slen):
            f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\n" % (args.pair, args.name, "mt", j, k, target[j][k], tgt_tags[i]))
            i += 1

with open(args.test_gap_output, "r") as f:
    gap_tags = []
    for line in f:
        gap_tags += line.rstrip().split(" ")
with open(os.path.join(args.output_dir, "predictions_gaps.txt"), "w") as f:
    i = 0
    for j, slen in enumerate(target_len):
        for k in range(slen + 1):
            f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\n" % (args.pair, args.name, "gap", j, k, "gap", gap_tags[i]))
            i += 1
