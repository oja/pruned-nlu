filename = 'snips/train/seq.in'


m = 0
with open(filename) as f:
    for line in f:
        line = line.rstrip().split()
        m = max(len(line), m)


print(m)
