import tqdm
import torch

for i in tqdm(range(config["n_bootstraps"])):
        # Generating random indices for sampling from dataframe
        N = len(y)
        torch.manual_seed(i)
        rng = np.random.default_rng(seed=i)
        idx = rng.choice(N, N, replace=True)