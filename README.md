# ExFlow
Explore Inter-layer Expert Affinity in MoE Model Inference

This is the repository for the main algorithm proposed in paper [https://arxiv.org/abs/2401.08383](https://arxiv.org/abs/2401.08383).

# Quick Start

This repo provides the integer programming model to solve the expert affinity. To run the solver, you need to first install `Gurobi` solver via the [official website](https://support.gurobi.com/hc/en-us).

For Python, you can easily install it via:
```
python -m pip install gurobipy
```

Note, that you will need a valid license to use the Gurobi solver, please follow the [official instruction](https://www.gurobi.com/academia/academic-program-and-licenses/) to get the license and place it in the right path.

Also, feel free to use other solvers if you are familiar with integer programming.


## 0. Implement context-coherence design in your MoE model
As described in our paper, you need to first ensure the context of all tokens are coherent across all GPUs. You can achieve this by calling `Allgather` operations in two places:

- 1. At the beginning of inference, make sure every GPU processing all prompts to get the initial context.
- 2. At the end of each forward, make sure the newly generated token(s) on each GPU are visible to all other GPUs.
- 3. Only keep the first `Alltoall` operation in your MoE module.

## 1. Get the expert routing trace log of your MoE model

ExFlow uses the expert routing trace log to profile the expert affinity, therefore, you need to save the trace log into a `.npy` file.

E.g.

```
trace.npy:
[number_of_tokens, number_of_layers]

[4, 1, 2, 7, 3, 1, 5, ...]
[0, 2, 6, 3, 5, 2, 1, ...]
[2, 2, 4, 1, 3, 6, 5, ...]
[5, 7, 1, 0, 3, 3, 6, ...]
...
```

The trace log should be a 2D numpy array, where each number denotes the index of the expert at each layer of each token.

## 2. Specify the number of GPUs to inference the MoE model
Then, we need to specify the following attributes. For expert parallelism, make sure `num_expert_per_layer` is divisible by `total_gpus`.
```
num_expert_per_layer = 8
intra_gpus = 4
nodes = 1
# total_gpus = intra_gpus * nodes
```

## 3. Specify number of tokens for profiling affinity
In the paper, we have analyzed in detail on how many tokens you typically need to get a precise profiling of the expert affinity. You can adjust this number.
```
incremental_amount = 5000
run_times = (num_tokens + incremental_amount - 1) // incremental_amount
```

Note that if you have to use a large number of tokens to profile at once, the complexity of the problem will be too heavy, therefore, we will use an incremental strategy to solve. Each time we only add a certain number of tokens to the solver, until we observe a steady solution. (Typically, 5000 tokens is enough)

## 4. Bipart Solver
Bipart strategy is a workaround to further accelerate the solver. Assume we want to inference the MoE model on 4 GPUs, we can either:

- a. Solve out the expert placement on 4 GPUs at once
- b. Partition 4 GPUs into two group, for example, [GPU 0, GPU 1] and [GPU 2, GPU 3]. We first solve the affinity for the two partitions. Then for the first partition [GPU 0, GPU 1], we further run a solver to finally get the expert placement on GPU 0 and GPU 1. Similar for the second partition. 

We found that even though the bipart strategy runs more solvers, but the search space for each solver is reduced exponentially. And it is able to find the near-optimal strategy easily.

## 5. Get the result
After the solver finishes, you will get an optimized expert placement table under `solutions`. For example, `solutions/solution_intra4_inter1_cost17.495347692691237_cintra5.831782564230412_cinter0.csv`

The file name denotes this is a solution for a single node with 4 GPUs. In every forward of the model, on each GPU, the average hop of token routing to other GPUs is `5.83`. The average hop of token routing between nodes is `0`, because we only have 1 node here.

Note that, in the `csv` file, we index the expert with a unique number. For example, if each layer has 8 experts, then in the first layer, we have experts `0, 1, 2, ..., 7`. In the second layer, we have experts `8, 9, 10, ..., 15`, etc.

## 6. Place the expert at the right GPU when launch the model
Finally, when you launch your MoE model, make sure to load the expert to the right GPU according to the solution you got in [step 5](#5-get-the-result).

# Example

## Mixtral 8x7B
We perform the profiling of expert affinity on the [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) model. The best expert placement strategies are given in this repo (under `solutions/`).
