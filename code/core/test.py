import re
fname = "results/kernel_nuts_results_1000_burnin_8192_epochs_2500_leapfrogsteps_512_nodes_[5, 20, 20, 1].txt"


# pattern = r"(\[.+?\]|[0-9]+[.]?[0-9]*|\w+)"
pattern = r"(\[.+?\]|\d+\.?\d*|\w+)"
with open(fname, "r") as infile:
    first_line = infile.readline()
    second_line = infile.readline()

print(repr(second_line))
matches = re.findall(pattern, second_line)
print(matches)
print(second_line)
for match in matches:
    print(match)
# print(re.findall(pattern, second_line))