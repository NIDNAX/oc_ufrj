import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import matplotlib.pyplot as plt

# Função para ler as instâncias do SCP
#=================================================================================
def read_set_cover_instance(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        m, n = map(int, first_line.split())
        
        costsList = []
        while len(costsList) < n:
            line = file.readline()
            if not line:
                break
            costsList += list(map(int, line.strip().split()))
        if len(costsList) != n:
            raise ValueError(f"Esperado {n} custos, mas encontrado {len(costsList)}")
        costs = {j: costsList[j-1] for j in range(1, n + 1)}
        
        subsets = {j: set() for j in range(1, n + 1)}
        for i in range(1, m + 1):
            numberElements_line = file.readline()
            if not numberElements_line:
                raise ValueError(f"Esperado número de elementos para item {i}, mas não encontrado.")
            numberElements = int(numberElements_line.strip())
            count = 0
            while count < numberElements:
                columns_line = file.readline()
                if not columns_line:
                    raise ValueError(f"Esperado mais colunas para item {i}, mas não encontrado.")
                columns = list(map(int, columns_line.strip().split()))
                for col in columns:
                    if col < 1 or col > n:
                        raise ValueError(f"Coluna {col} inválida para item {i} com n={n}")
                    subsets[col].add(i)
                    count += 1
                    if count >= numberElements:
                        break
        
        U = set(range(1, m + 1))
        return U, costs, subsets
#=================================================================================

# Heurística gulosa para encontrar uma solução viável inicial
#=================================================================================
def greedy_set_cover(U, subsets, costs):
    uncovered = set(U)
    selected_subsets = set()
    total_cost = 0
    while uncovered:
        best_subset = None
        best_cost_effectiveness = float('inf')
        for j in subsets:
            covered = subsets[j] & uncovered
            if not covered:
                continue
            cost_effectiveness = costs[j] / len(covered)
            if cost_effectiveness < best_cost_effectiveness:
                best_cost_effectiveness = cost_effectiveness
                best_subset = j
        if best_subset is None:
            return None, float('inf')
        selected_subsets.add(best_subset)
        total_cost += costs[best_subset]
        uncovered -= subsets[best_subset]
    return selected_subsets, total_cost
#=================================================================================

# Inicializa multiplicadores lagrangeanos
#=================================================================================
def initialize_lagrange_multipliers(U):
    return {i: 0.0 for i in U}

# Calcula c′_j = c_j − Somatório (i ∈ J_j) u_i
def compute_reduced_costs(costs, subsets, multipliers, n):
    reduced_costs = {}
    for j in range(1, n + 1):
        reduced_costs[j] = costs[j] - sum(multipliers[i] for i in subsets[j])
    return reduced_costs
#=================================================================================

# Função objetivo Lagrangeana
#=================================================================================
def lagrange_objective(reduced_costs, multipliers, U):
    objective_value = sum(min(0, reduced_costs[j]) for j in reduced_costs) + sum(multipliers[i] for i in U)
    return objective_value
#=================================================================================

# Resolve o subproblema Lagrangeano
#=================================================================================
def solve_lagrange_subproblem(reduced_costs):
    solution = {j: 1 if reduced_costs[j] < 0 else 0 for j in reduced_costs}
    return solution

# Calcula o subgradiente
def compute_subgradient(U, subsets, solution):
    subgrad = {}
    for i in U:
        covered = sum(solution[j] for j in subsets if i in subsets[j])
        subgrad[i] = 1 - covered
    return subgrad
#=================================================================================

# Atualiza os multiplicadores usando o subgradiente
#=================================================================================
def update_multipliers(multipliers, subgrad, T, epsilon):
    for i in multipliers:
        multipliers[i] = max(0, multipliers[i] + (1 + epsilon) * T * subgrad[i])
    return multipliers
#=================================================================================

# Algoritmo do subgradiente
#=================================================================================
def solve_set_cover_lagrange(U, costs, subsets, ub, max_iterations=1000, tolerance=1e-5, epsilon=0.02, pi_initial=2.0, pi_reduction_threshold=10):
    multipliers = initialize_lagrange_multipliers(U)
    best_z = -float('inf')
    best_solution = None
    pi = pi_initial
    no_improvement_steps = 0
    
    upper_bounds = []
    lower_bounds = []
    
    for k in range(1, max_iterations + 1):
        reduced_costs = compute_reduced_costs(costs, subsets, multipliers, len(costs))
        solution = solve_lagrange_subproblem(reduced_costs)
        z_u = lagrange_objective(reduced_costs, multipliers, U)
        
        lower_bounds.append(z_u)
        
        selected_subsets, greedy_cost = greedy_set_cover(U, subsets, costs)
        if greedy_cost < ub:
            ub = greedy_cost
        
        upper_bounds.append(ub)
        
        if z_u > best_z:
            best_z = z_u
            best_solution = solution
            no_improvement_steps = 0
            print(f"Iteração {k}: Melhor z(u) = {best_z}")
        else:
            no_improvement_steps += 1
            print(f"Iteração {k}: z(u) = {z_u}, sem melhora")
        
        if (ub - best_z) < tolerance:
            print(f"Terminando: ub - z(u) = {ub - best_z} < tolerance na iteração {k}")
            break
        
        if no_improvement_steps >= pi_reduction_threshold:
            pi = pi / 2
            print(f"Reduzindo pi para {pi} na iteração {k}")
            no_improvement_steps = 0
        
        subgrad = compute_subgradient(U, subsets, solution)
        norm_subgrad = sum(g**2 for g in subgrad.values())
        
        if norm_subgrad == 0:
            print(f"Terminando: norma do subgradiente é zero na iteração {k}")
            break
        
        T = pi * (ub - z_u) / (norm_subgrad + 1e-10)
        
        multipliers = update_multipliers(multipliers, subgrad, T, epsilon)
    
    return best_z, best_solution, upper_bounds, lower_bounds
#=================================================================================

# Resolver usando CPLEX
#=================================================================================
def solve_set_cover_cplex(U, costs, subsets):
    try:
        cpx = Cplex()
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_results_stream(None)
        
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        
        num_columns = len(costs)
        cpx.variables.add(obj=[costs[j] for j in range(1, num_columns + 1)],
                          types=[cpx.variables.type.binary] * num_columns,
                          names=[f'x_{j}' for j in range(1, num_columns + 1)])
        
        for i in U:
            J_i = [j for j in subsets if i in subsets[j]]
            if not J_i:
                print(f"Item {i} não é coberto por nenhum subconjunto.")
                return None, float('inf')
            J_i_indices = [j - 1 for j in J_i]
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=J_i_indices, val=[1.0] * len(J_i_indices))],
                senses=["G"],
                rhs=[1.0]
            )
        
        cpx.solve()
        
        solution = cpx.solution.get_values()
        objective_value = cpx.solution.get_objective_value()
        
        cplex_solution = {j: int(round(solution[j-1])) for j in range(1, num_columns + 1)}
        
        return cplex_solution, objective_value
    
    except CplexError as e:
        print(f"Erro no CPLEX: {e}")
        return None, float('inf')
#=============================================================================================

# Função para comparar soluções Lagrangeanas com a solução obtida pelo CPLEX
#=================================================================================
def compare_solutions(lagrange_solution, cplex_solution):
    matches = sum(1 for j in lagrange_solution if lagrange_solution[j] == cplex_solution[j])
    total = len(lagrange_solution)
    accuracy = matches / total * 100
    print(f"Acurácia da solução lagrangeana em comparação ao CPLEX: {accuracy:.2f}%")
    return accuracy
#=================================================================================

# Geração do gráfico de comparação
#=================================================================================
def plot_bounds_comparison(upper_bounds, lower_bounds, cplex_optimal, instance_name):
    iterations = range(1, len(upper_bounds) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, upper_bounds, label="Upper Bound", color="red")
    plt.plot(iterations, lower_bounds, label="Lower Bound", color="blue")
    plt.axhline(y=cplex_optimal, color='green', linestyle='--', label="Ótimo (CPLEX)")
    
    plt.xlabel("Iterações")
    plt.ylabel("Valor da Função Objetivo")
    plt.title(f"Comparação de Upper Bound e Lower Bound - {instance_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
#=================================================================================

# Main - Execução do SCP com Relaxação Lagrangeana e comparação com CPLEX
#=================================================================================
if __name__ == "__main__":
    # Leitura da instância
    U, costs, subsets = read_set_cover_instance(".\INSTANCES\scp41.txt")

    # Valor ótimo via CPLEX
    _, cplex_optimal = solve_set_cover_cplex(U, costs, subsets)
    
    # Resolvendo via Relaxação Lagrangeana
    initial_ub = float('inf')
    best_z, best_solution, upper_bounds, lower_bounds = solve_set_cover_lagrange(U, costs, subsets, initial_ub)

    # Comparação entre as soluções Lagrangeana e CPLEX
    cplex_solution, _ = solve_set_cover_cplex(U, costs, subsets)
    compare_solutions(best_solution, cplex_solution)
    
    # Geração do gráfico de comparação
    plot_bounds_comparison(upper_bounds, lower_bounds, cplex_optimal, "Instância de Teste")
#=================================================================================

