import numpy as np
from gurobipy import Model, GRB, quicksum
import csv
import random
import math
from copy import deepcopy
import os

routing_array = np.load('expert_trace/your_trace.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
num_tokens, _ = routing_array.shape
print(num_tokens)
num_layer = routing_array.shape[1]
num_expert_per_layer = 8
total_experts = num_expert_per_layer * num_layer
assert total_experts % 2 == 0
intra_gpus = 8
nodes = 1
use_bipart = True
incremental_amount = 5000
run_times = (num_tokens + incremental_amount - 1) // incremental_amount
time_limits = 1800 if not use_bipart else 60

for i in range(num_layer):
    routing_array[:, i] += num_expert_per_layer * i
print(routing_array)

def calculate_cost(message, solution, intra_gpus=0):
    cost = 0
    count = 0
    intra = 0
    inter = 0
    for i in range(len(message)-1):
        if intra_gpus == 0:
            if message[i] in solution.keys() and message[i+1] in solution.keys():
                count += 1
                cost += abs(solution[message[i]] - solution[message[i+1]])
        else:
            count += 1
            node_a = solution[message[i]] // intra_gpus
            node_b = solution[message[i+1]] // intra_gpus
            if solution[message[i]] != solution[message[i+1]]:
                if node_a == node_b:
                    cost += 1
                    intra += 1
                else:
                    cost += 2
                    inter += 1

    if intra_gpus != 0:
        return cost / (count + 1e-8), intra / (count + 1e-8), inter / (count + 1e-8)
    else:
        return cost / (count + 1e-8)


def solve_graph_optimization(fused_obj=False, cur_partition_experts=None, cur_partition_balance_experts_per_layer=0, iters=0, time_limit=60, hard_limit=0.0, partitions=2):

    increment = incremental_amount
    solution_storage = []

    layered_experts_dict = {}
    for i in range(num_layer):
        layered_experts_dict[i] = []

    for k in cur_partition_experts:
        layered_experts_dict[int(k) // num_expert_per_layer].append(k)

    for i in range(increment, int(iters*increment) + 1, increment):
        i = num_tokens if i > num_tokens else i
        subset_messages = routing_array[: i]

        m = Model()
        m.Params.TimeLimit = time_limit
        m.setParam('Heuristics', 1.0)

        x = {}
        for n in cur_partition_experts:
            for c in range(partitions):
                x[n, c] = m.addVar(vtype=GRB.BINARY, name=f'x_{n}_{c}')

        cost = {}
        for k in range(i):
            for s in range(num_layer - 1):
                cost[k, s] = m.addVar(vtype=GRB.BINARY, name=f'cost_{k}_{s}')

        load_balance = {}
        for layer_idx in range(num_layer):
            for c in range(partitions):
                load_balance[layer_idx, c] = m.addVar(lb=-cur_partition_balance_experts_per_layer//2, ub=cur_partition_balance_experts_per_layer//2+1, vtype=GRB.INTEGER, name=f'loadBalanceValue_{layer_idx}_{c}')

        expert_per_layer_per_node_abs = {}
        for l in range(num_layer):
            for n in range(partitions):
                expert_per_layer_per_node_abs[l, n] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"abs_{l}_{n}")

        m.update()
        for layer_idx in range(num_layer):
            for c in range(partitions):
                m.addConstr(load_balance[layer_idx, c] == (sum(x[n, c] for n in layered_experts_dict[layer_idx]) - cur_partition_balance_experts_per_layer), name=f"loadConstr_{layer_idx}_{c}")

        m.update()
        for l in range(num_layer):
            for n in range(partitions):
                m.addGenConstrAbs(expert_per_layer_per_node_abs[l, n], load_balance[l, n], name=f"absConstr_{l}_{n}")

        if not fused_obj:
            m.update()
            for layer_idx in range(num_layer):
                for c in range(partitions):
                    if hard_limit != 0.0:
                        m.addConstr(
                            quicksum(x[n, c] for n in layered_experts_dict[layer_idx]) >= math.ceil(cur_partition_balance_experts_per_layer * hard_limit), name=f"hardLoadBalance_{layer_idx}_{c}")
                    else:
                        m.addConstr(
                            quicksum(x[n, c] for n in layered_experts_dict[layer_idx]) == cur_partition_balance_experts_per_layer, name=f"hardLoadBalance_{layer_idx}_{c}")

        m.update()
        # m.addConstr(x[cur_partition_experts[0], 0] == 1)
        for n in cur_partition_experts:
            m.addConstr(quicksum(x[n, c] for c in range(partitions)) == 1, name=f"allone_{n}")

        if hard_limit != 0.0:
            # Each partition has P/N nodes
            for c in range(partitions):
                m.addConstr(quicksum(x[n, c] for n in cur_partition_experts) == int(cur_partition_balance_experts_per_layer * num_layer), name=f"allequal_{c}")

        count_valid_step = 0
        for k in range(i):
            for s in range(num_layer - 1):
                if (subset_messages[k][s] in cur_partition_experts) and (subset_messages[k][s + 1] in cur_partition_experts):
                    count_valid_step += 1
                    for c in range(partitions):
                        m.addConstr(cost[k, s] >= x[subset_messages[k][s], c] - x[subset_messages[k][s + 1], c], name=f"costConstr1_{k}_{s}")
                        m.addConstr(cost[k, s] >= x[subset_messages[k][s + 1], c] - x[subset_messages[k][s], c], name=f"costConstr2_{k}_{s}")

        # Objective
        if fused_obj:
            m.setObjective(quicksum(cost[k, s] for k in range(i) for s in range(num_layer - 1)) / count_valid_step + 1 / num_layer / cur_partition_balance_experts_per_layer * quicksum(expert_per_layer_per_node_abs[l, n] for l in range(num_layer) for n in range(partitions)), GRB.MINIMIZE)
        else:
            m.setObjective(quicksum(cost[k, s] for k in range(i) for s in range(num_layer - 1)) / count_valid_step, GRB.MINIMIZE)

        m.update()

        if i//increment > 1:
            for n, c in solution_storage[-1].items():
                m.getVarByName(f"x_{n}_{c}").start = 1.0
            print("Loading solutions...")

        m.optimize()

        if m.SolCount > 0:
            solution = {}
            for n in cur_partition_experts:
                for c in range(partitions):
                    if x[n, c].x > 0.5:
                        solution[n] = c
            print(f"Complete {i/num_tokens*100}%: {solution}")
            solution_storage.append(solution)

            avg_cost = sum(calculate_cost(message, solution) for message in routing_array) / num_tokens
            load_balance_output = {}
            for layer_idx in range(num_layer):
                for c in range(partitions):
                    load_balance_output[layer_idx, c] = load_balance[layer_idx, c].x

            print(f"Complete {i/num_tokens*100}%, Average cost per token: {avg_cost}, load balance max: {np.max(np.array(list(load_balance_output.values())))}, load balance min: {np.min(np.array(list(load_balance_output.values())))}, load balance stdv: {np.std(np.array(list(load_balance_output.values())))}")
        else:
            print("No solution found.")
        del m

    return solution_storage[-1]

def vanilla_placement(num_layer, num_expert_per_layer, intra_gpus, nodes):
    overall_gpus = intra_gpus * nodes
    expert_per_gpu = num_expert_per_layer / overall_gpus
    placement = {}
    for i in range(num_layer * num_expert_per_layer):
        placement[i] = int((i // expert_per_gpu) % overall_gpus)
    return placement

def read_parition(total_number_gpu, use_bipart=False):
    solution_dict = {}
    cur_partition_experts = [i for i in range(num_layer * num_expert_per_layer)]

    if use_bipart:
        cur_partition_balance_experts_per_layer = num_expert_per_layer // 2
        total_level = int(np.log2(total_number_gpu))
        for i in range(total_level):
            if i == 0:
                cur_solution = solve_graph_optimization(False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=2)
                upper_level_solution = deepcopy(cur_solution)
            else:
                num_problems = 2 ** i
                for j in range(num_problems):
                    cur_partition_experts = []
                    for k, v in upper_level_solution.items():
                        if v == j:
                            cur_partition_experts.append(k)
                    sub_solution = solve_graph_optimization(False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=2)
                    for k, v in sub_solution.items():
                        cur_solution[k] = cur_solution[k] * 2 + v
                upper_level_solution = deepcopy(cur_solution)

            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // 2

        cur_solution = upper_level_solution

    else:
        cur_partition_balance_experts_per_layer = num_expert_per_layer // nodes
        print(len(cur_partition_experts))

        if nodes > 1:
            cur_solution = solve_graph_optimization(False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=nodes)
            upper_level_solution = deepcopy(cur_solution)
            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // intra_gpus

            for i in range(nodes):
                cur_partition_experts = []
                partition_offset = i * intra_gpus
                for k, v in upper_level_solution.items():
                    if v == i:
                        cur_partition_experts.append(k)
                print(len(cur_partition_experts))
                sub_solution = solve_graph_optimization(False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=intra_gpus)
                for k, v in sub_solution.items():
                    cur_solution[k] = v + partition_offset
        else:
            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // intra_gpus
            cur_solution = solve_graph_optimization(False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=intra_gpus)
    
    ######################
    vanilla_p = vanilla_placement(num_layer, num_expert_per_layer, intra_gpus, nodes)
    print(vanilla_p)
    print("Vanilla placement:")
    cost = 0
    intra = 0
    inter = 0
    for message in routing_array:
        cost_, intra_, inter_ = calculate_cost(message, vanilla_p, intra_gpus=intra_gpus)
        cost += cost_
        intra += intra_
        inter += inter_
    avg_cost = cost / num_tokens * (num_layer - 1)
    avg_intra = intra / num_tokens * (num_layer - 1) / (intra_gpus - 1)
    if nodes > 1:
        avg_inter = inter / num_tokens * (num_layer - 1) / (intra_gpus * (nodes - 1))
    else:
        avg_inter = 0
        
    print(avg_cost, avg_intra, avg_inter)

    #######################
    print(cur_solution)
    cost = 0
    intra = 0
    inter = 0
    for message in routing_array:
        cost_, intra_, inter_ = calculate_cost(message, cur_solution, intra_gpus=intra_gpus)
        cost += cost_
        intra += intra_
        inter += inter_
    avg_cost = cost / num_tokens * (num_layer - 1)
    avg_intra = intra / num_tokens * (num_layer - 1) / (intra_gpus - 1)
    if nodes > 1:
        avg_inter = inter / num_tokens * (num_layer - 1) / (intra_gpus * (nodes - 1))
    else:
        avg_inter = 0

    file_name = f'solutions/solution_intra{intra_gpus}_inter{nodes}_cost{avg_cost}_cintra{avg_intra}_cinter{avg_inter}.csv'

    with open(os.path.join(file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Expert", "Node"])
        for expert, node in cur_solution.items():
            writer.writerow([expert, node])

    print("After optimization:")
    print(avg_cost, avg_intra, avg_inter)



read_parition(intra_gpus * nodes, use_bipart)
