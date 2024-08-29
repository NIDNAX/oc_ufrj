import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError

# Função para ler as instâncias do SCP
def read_set_cover_instance(filename):
    with open(filename, 'r') as file:
        # Linha 1: número de linhas (m) e número de colunas (n)
        first_line = file.readline().strip()
        m, n = map(int, first_line.split())
        
        # Ler os custos c(j) de cada coluna j nas próximas linhas até que n valores sejam lidos
        costsList = []
        while len(costsList) < n:
            line = file.readline()
            if not line:
                break
            costsList += list(map(int, line.strip().split()))
        if len(costsList) != n:
            raise ValueError(f"Esperado {n} custos, mas encontrado {len(costsList)}")
        costs = {j: costsList[j-1] for j in range(1, n + 1)}
        
        # Inicialização do dicionário de subconjuntos que cobrem cada item
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
        
        # Conjunto universal
        U = set(range(1, m + 1))
        return U, costs, subsets

# Heurística gulosa para encontrar uma solução viável inicial
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
            # Nenhuma solução viável encontrada
            return None, float('inf')
        selected_subsets.add(best_subset)
        total_cost += costs[best_subset]
        uncovered -= subsets[best_subset]
    return selected_subsets, total_cost

# Inicializa multiplicadores lagrangeanos
def initialize_lagrange_multipliers(U):
    return {i: 0.0 for i in U}

# Calcula c′_j = c_j − Somatório (i ∈ J_j) u_i
def compute_reduced_costs(costs, subsets, multipliers, n):
    reduced_costs = {}
    for j in range(1, n + 1):
        reduced_costs[j] = costs[j] - sum(multipliers[i] for i in subsets[j])
    return reduced_costs

# Função objetivo Lagrangeana
def lagrange_objective(reduced_costs, multipliers, U):
    # Calcula sum_j min(0, c'_j) + sum_i u_i
    objective_value = sum(min(0, reduced_costs[j]) for j in reduced_costs) + sum(multipliers[i] for i in U)
    return objective_value

# Resolve o subproblema Lagrangeano
def solve_lagrange_subproblem(reduced_costs):
    # Seleciona x_j = 1 se c'_j < 0, caso contrário x_j = 0
    solution = {j: 1 if reduced_costs[j] < 0 else 0 for j in reduced_costs}
    return solution

# Calcula o subgradiente
def compute_subgradient(U, subsets, solution):
    subgrad = {}
    for i in U:
        # Somatório de x_j para j ∈ J_i
        covered = sum(solution[j] for j in subsets if i in subsets[j])
        subgrad[i] = 1 - covered
    return subgrad

# Atualiza os multiplicadores usando o subgradiente
def update_multipliers(multipliers, subgrad, T, epsilon):
    for i in multipliers:
        multipliers[i] = max(0, multipliers[i] + (1 + epsilon) * T * subgrad[i])
    return multipliers

# Algoritmo do subgradiente
def solve_set_cover_lagrange(U, costs, subsets, ub, max_iterations=1000, tolerance=1e-5, epsilon=0.02, pi_initial=2.0, pi_reduction_threshold=10):
    multipliers = initialize_lagrange_multipliers(U)
    best_z = -float('inf')  # z(u) é para ser maximizado
    best_solution = None
    pi = pi_initial
    no_improvement_steps = 0
    last_best_z = -float('inf')
    
    for k in range(1, max_iterations + 1):
        reduced_costs = compute_reduced_costs(costs, subsets, multipliers, len(costs))
        solution = solve_lagrange_subproblem(reduced_costs)
        z_u = lagrange_objective(reduced_costs, multipliers, U)
        
        # Atualizar best_z e best_solution
        if z_u > best_z:
            best_z = z_u
            best_solution = solution
            # Reset no_improvement_steps
            no_improvement_steps = 0
            print(f"Iteração {k}: Melhor z(u) = {best_z}")
        else:
            no_improvement_steps += 1
            print(f"Iteração {k}: z(u) = {z_u}, sem melhora")
        
        # Verificar condição de parada
        if (ub - best_z) < tolerance:
            print(f"Terminando: ub - z(u) = {ub - best_z} < tolerance na iteração {k}")
            break
        
        # Verificar se pi precisa ser reduzido
        if no_improvement_steps >= pi_reduction_threshold:
            pi = pi / 2
            print(f"Reduzindo pi para {pi} na iteração {k}")
            no_improvement_steps = 0
        
        # Calcular subgradiente
        subgrad = compute_subgradient(U, subsets, solution)
        norm_subgrad = sum(g**2 for g in subgrad.values())
        
        if norm_subgrad == 0:
            print(f"Terminando: norma do subgradiente é zero na iteração {k}")
            break
        
        # Calcular tamanho de passo T
        T = pi * (ub - z_u) / (norm_subgrad + 1e-10)  # Adiciona uma constante para evitar divisão por zero
        
        # Atualizar multiplicadores
        multipliers = update_multipliers(multipliers, subgrad, T, epsilon)
    
    return best_z, best_solution


# Resolver usando CPLEX
#=================================================================================
def solve_set_cover_cplex(U, costs, subsets):
    try:
        cpx = Cplex()
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_results_stream(None)
        
        # Definir o problema como minimização
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        
        # Adicionar variáveis binárias
        num_columns = len(costs)
        cpx.variables.add(obj=[costs[j] for j in range(1, num_columns + 1)],
                          types=[cpx.variables.type.binary] * num_columns,
                          names=[f'x_{j}' for j in range(1, num_columns + 1)])
        
        # Adicionar restrições: para cada i, Somatório x_j para j ∈ J_i ≥ 1
        for i in U:
            J_i = [j for j in subsets if i in subsets[j]]
            if not J_i:
                # Item i não é coberto por nenhum subconjunto
                print(f"Item {i} não é coberto por nenhum subconjunto.")
                return None, float('inf')
            # CPLEX usa índices 0-based
            J_i_indices = [j - 1 for j in J_i]
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=J_i_indices, val=[1.0] * len(J_i_indices))],
                senses=["G"],
                rhs=[1.0]
            )
        
        # Resolver o problema
        cpx.solve()
        
        # Extrair a solução
        solution = cpx.solution.get_values()
        objective_value = cpx.solution.get_objective_value()
        
        # Converter a solução para um dicionário {j: 0 ou 1}
        cplex_solution = {j: int(round(solution[j-1])) for j in range(1, num_columns + 1)}
        
        return cplex_solution, objective_value
    
    except CplexError as e:
        print(f"Erro no CPLEX: {e}")
        return None, float('inf')
#=============================================================================================




""" # Função para comparar soluções Lagrangeana e CPLEX
def compare_solutions(U, costs, subsets, filename):
    # Encontrar uma solução viável inicial usando heurística
    feasible_solution, feasible_cost = greedy_set_cover(U, subsets, costs)
    if feasible_solution is None:
        print("Nenhuma solução viável encontrada pela heurística.")
        ub = float('inf')
    else:
        print(f"Solução inicial heurística: {feasible_solution}")
        print(f"Custo da solução inicial: {feasible_cost}")
        ub = feasible_cost
    
    # Resolver usando relaxação lagrangeana
    if ub < float('inf'):
        print("\nIniciando a solução via Relaxação Lagrangeana...")
        z_lower_bound, lagrange_solution = solve_set_cover_lagrange(U, costs, subsets, ub)
        print(f"Valor objetivo (Relaxação Lagrangeana - Lower Bound): {z_lower_bound}")
        print(f"Solução Relaxação Lagrangeana (Dual): {lagrange_solution}")
    else:
        print("Não é possível executar a relaxação lagrangeana sem um limite superior (ub).")
        z_lower_bound = -float('inf')
    
    # Resolver usando CPLEX
    print("\nIniciando a solução via CPLEX...")
    cplex_solution, cplex_value = solve_set_cover_cplex(U, costs, subsets)
    if cplex_solution is not None:
        print(f"Solução CPLEX: {cplex_solution}")
        print(f"Valor objetivo (CPLEX): {cplex_value}")
    else:
        print("Não foi possível obter a solução via CPLEX.")
    
    # Calcular e mostrar o gap
    if z_lower_bound > -float('inf') and cplex_value < float('inf'):
        gap = cplex_value - z_lower_bound
        print(f"\nGap (CPLEX - Relaxação Lagrangeana): {gap}")
    else:
        print("Não foi possível calcular o gap devido a soluções ausentes.") """


# Exemplo de execução
if __name__ == "__main__":
    instance_file = "./INSTANCES/scp43.txt"
    U, costs, subsets = read_set_cover_instance(instance_file)
    

    # Encontrar uma solução viável inicial
    feasible_solution, feasible_cost = greedy_set_cover(U, subsets, costs)
    if feasible_solution is not None:
        print(f"Solução inicial heurística: {feasible_solution}")
        print(f"Custo da solução inicial: {feasible_cost}")
        ub = feasible_cost
    else:
        print("Nenhuma solução viável encontrada pela heurística.")
        ub = float('inf')
    
    # Resolver usando relaxação lagrangeana
    if ub < float('inf'):
        print("\nIniciando a solução via Relaxação Lagrangeana...")
        z_lower_bound, lagrange_solution = solve_set_cover_lagrange(U, costs, subsets, ub)
        print(f"Solução Relaxação Lagrangeana (Dual): {lagrange_solution}")
        print(f"Valor objetivo (Relaxação Lagrangeana - Lower Bound): {z_lower_bound}")
    else:
        print("Não é possível executar a relaxação lagrangeana sem um limite superior (ub).")
        z_lower_bound = -float('inf')
    
    # Resolver usando CPLEX
    print("\nIniciando a solução via CPLEX...")
    cplex_solution, cplex_value = solve_set_cover_cplex(U, costs, subsets)
    if cplex_solution is not None:
        print(f"Solução CPLEX: {cplex_solution}")
        print(f"Valor objetivo (CPLEX): {cplex_value}")
    else:
        print("Não foi possível obter a solução via CPLEX.")
    
    # Calcular e mostrar o gap
    if z_lower_bound > -float('inf') and cplex_value < float('inf'):
        gap = cplex_value - z_lower_bound
        print(f"\nGap (CPLEX - Relaxação Lagrangeana): {gap}")
    else:
        print("Não foi possível calcular o gap devido a soluções ausentes.")
