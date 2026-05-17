import numpy as np
import mmengine

lines = []
with open('train_100.txt', 'r') as f:
    for line in f:
        lines.append(line.strip())


np.random.seed(0)
np.random.shuffle(lines)

for i in [20, 40, 60, 80]:
    n_lines = i * len(lines) // 100

    subset = lines[: n_lines]
    with open('pretrain_%d.txt'%(i), 'w') as f:
        for l in subset:
            f.write(l + '\n')

    subset = lines[-n_lines:]
    with open('finetune_%d.txt'%(i), 'w') as f:
        for l in subset:
            f.write(l + '\n')
    
