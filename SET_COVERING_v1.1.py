import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import matplotlib.pyplot as plt
import os
import time

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

# Heurística gulosa básica para o SCP
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

# Heurística Gulosa Adaptada para o Algoritmo do Subgradiente
#=================================================================================
def adaptive_greedy_set_cover(U, subsets, costs, dual_solution):
    Jk = {j for j in dual_solution if dual_solution[j] > 0}
    J_star = set()
    
    uncovered = set(U)
    
    # Fase 1: Aplicar a heurística restrita a J^k
    while uncovered and Jk:
        best_subset = None
        best_cost_effectiveness = float('inf')
        for j in Jk:
            covered = subsets[j] & uncovered
            if not covered:
                continue
            denominator = len(covered - {i for i in uncovered if i in J_star})
            if denominator > 0:
                cost_effectiveness = costs[j] / denominator
                if cost_effectiveness < best_cost_effectiveness:
                    best_cost_effectiveness = cost_effectiveness
                    best_subset = j
        if best_subset is None:
            break
        J_star.add(best_subset)
        uncovered -= subsets[best_subset]
        Jk.remove(best_subset)
    
    # Fase 2: Completar a heurística com J \ J^k se necessário
    if uncovered:
        remaining_subsets = set(subsets.keys()) - J_star
        while uncovered:
            best_subset = None
            best_cost_effectiveness = float('inf')
            for j in remaining_subsets:
                covered = subsets[j] & uncovered
                if not covered:
                    continue
                cost_effectiveness = costs[j] / len(covered)
                if cost_effectiveness < best_cost_effectiveness:
                    best_cost_effectiveness = cost_effectiveness
                    best_subset = j
            if best_subset is None:
                break
            J_star.add(best_subset)
            uncovered -= subsets[best_subset]
            remaining_subsets.remove(best_subset)
    
    total_cost = sum(costs[j] for j in J_star)
    return J_star, total_cost

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
    contador_pi=0
    upper_bounds = []
    lower_bounds = []
    printar_testes= False
    for k in range(1, max_iterations + 1):
        reduced_costs = compute_reduced_costs(costs, subsets, multipliers, len(costs))
        solution = solve_lagrange_subproblem(reduced_costs)
        z_u = lagrange_objective(reduced_costs, multipliers, U)
        
        lower_bounds.append(z_u)
        
        # Aplicar a heurística gulosa adaptada
        J_star, greedy_cost = adaptive_greedy_set_cover(U, subsets, costs, solution)
        if greedy_cost < ub:
            ub = greedy_cost
        
        upper_bounds.append(ub)
        
        if z_u > best_z:
            best_z = z_u
            best_solution = solution
            no_improvement_steps = 0
            if printar_testes == True:
                print(f"Iteração {k}: Melhor z(u) = {best_z}")
        else:
            no_improvement_steps += 1
            if printar_testes == True:
                print(f"Iteração {k}: z(u) = {z_u}, sem melhora")
        
        if (ub - best_z) < tolerance:
            if printar_testes == True:
                print(f"Terminando: ub - z(u) = {ub - best_z} < tolerance na iteração {k}")
            break
        
        if no_improvement_steps >= pi_reduction_threshold:
            pi = pi / 2
            if printar_testes == True:
                print(f"Reduzindo pi para {pi} na iteração {k}")
            contador_pi+=1
            no_improvement_steps = 0
        
        subgrad = compute_subgradient(U, subsets, solution)
        norm_subgrad = sum(g**2 for g in subgrad.values())
        
        if norm_subgrad == 0:
            if printar_testes == True:
                print(f"Terminando: norma do subgradiente é zero na iteração {k}")
            break
        
        T = pi * (ub - z_u) / (norm_subgrad + 1e-10)
        
        multipliers = update_multipliers(multipliers, subgrad, T, epsilon)
    
    return best_z, best_solution, upper_bounds, lower_bounds, contador_pi
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
#=================================================================================

# Função principal para executar os algoritmos
#=================================================================================
def main(instance_file):
    U, costs, subsets = read_set_cover_instance(instance_file)
    
    print("Resolvendo SCP com Relaxação Lagrangeana:")
    best_z, best_solution, upper_bounds, lower_bounds = solve_set_cover_lagrange(U, costs, subsets, float('inf'))
    print(f"Melhor valor Z: {best_z}")
    
    print("Resolvendo SCP com CPLEX:")
    cplex_value, cplex_solution = solve_set_cover_cplex(U, costs, subsets)
    print(f"Valor ótimo CPLEX: {cplex_value}")
    

# Função para comparar soluções Lagrangeanas com a solução obtida pelo CPLEX
#=================================================================================
def compare_solutions(lagrange_solution, cplex_solution, cplex_optimal,lagrangeano, primal):
    matches = sum(1 for j in lagrange_solution if lagrange_solution[j] == cplex_solution[j])
    total = len(lagrange_solution)
    accuracy = matches / total * 100
    GAP_Primal = round(((min(primal)-(cplex_optimal)))/(cplex_optimal)*100,2)
    GAP_Dual = round(((cplex_optimal)-(max(lagrangeano)))/(cplex_optimal)*100,2)
    #print(f"Acurácia da solução lagrangeana em comparação ao CPLEX: {accuracy:.2f}%")
    return GAP_Dual, accuracy, GAP_Primal
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

instances=["scp41.txt","scp42.txt","scp43.txt","scp44.txt","scp45.txt","scp46.txt","scp47.txt","scp48.txt","scp49.txt","scp410.txt",
           "scp51.txt","scp52.txt","scp53.txt","scp54.txt","scp55.txt","scp56.txt","scp57.txt","scp58.txt","scp59.txt","scp510.txt",
           "scp61.txt","scp62.txt","scp63.txt","scp64.txt","scp65.txt",
           "scpa1.txt","scpa2.txt","scpa3.txt","scpa4.txt","scpa5.txt",
           "scpb1.txt","scpb2.txt","scpb3.txt","scpb4.txt","scpb5.txt",
           "scpc1.txt","scpc2.txt","scpc3.txt","scpc4.txt","scpc5.txt",
           "scpd1.txt","scpd2.txt","scpd3.txt","scpd4.txt","scpd5.txt",
           "scpe1.txt","scpe2.txt","scpe3.txt","scpe4.txt","scpe5.txt",
           "scpnre1.txt", "scpnre2.txt", "scpnre3.txt", "scpnre4.txt", "scpnre5.txt",
           "scpnrf1.txt", "scpnrf2.txt", "scpnrf3.txt", "scpnrf4.txt", "scpnrf5.txt",
           "scpnrg1.txt", "scpnrg2.txt", "scpnrg3.txt", "scpnrg4.txt", "scpnrg5.txt",
           "scpnrh1.txt", "scpnrh2.txt", "scpnrh3.txt", "scpnrh4.txt", "scpnrh5.txt"]



for i in instances:
    if __name__ == "__main__":
        # Leitura da instância
        instancia = i
        caminho_arquivo = os.path.join("..", "INSTANCES", instancia)
        print("Lendo Instâncias...")
        U, costs, subsets = read_set_cover_instance(caminho_arquivo)
        
        # Valor ótimo via CPLEX
        _, cplex_optimal = solve_set_cover_cplex(U, costs, subsets)
        
        # Resolvendo via Relaxação Lagrangeana
        initial_ub = float('inf')
        print("Resolvendo SCP via Relaxação Lagrangeana...")
        time_inicio_RL=time.time()
        best_z, best_solution, upper_bounds, lower_bounds, contador_pi = solve_set_cover_lagrange(U, costs, subsets, initial_ub)
        time_fim_RL=time.time()
        time_RL=time_fim_RL-time_inicio_RL    

        # Comparação entre as soluções Lagrangeana e CPLEX
        print("Resolvendo SCP via CPLEX...")
        time_inicio_CPLEX=time.time()
        cplex_solution, _ = solve_set_cover_cplex(U, costs, subsets)
        time_fim_CPLEX=time.time()
        time_CPLEX=time_fim_CPLEX-time_inicio_CPLEX
        #compare_solutions(best_solution, cplex_solution, cplex_optimal,lower_bounds)

        GAP_Dual, precisão, GAP_Primal = compare_solutions(best_solution, cplex_solution,cplex_optimal,lower_bounds,upper_bounds)
        GAP= round(GAP_Primal + GAP_Dual,2)

        print("=========================================")
        print("INSTÂNCIA: ", i)
        print("=========================================")
        print("ÓTIMO CPLEX: ", cplex_optimal)
        print("RELAXAÇÃO LAGRANGEANA: ", max(lower_bounds))
        print("PRECISÃO DO MÉTODO: ", precisão,"%")
        print("=========================================")
        print("GAP: ", GAP,"%")
        print("GAP Primal: ", GAP_Primal,"%")
        print("GAP Dual: ", GAP_Dual,"%")
        print("=========================================")
        print("Tempo de Execução CPLEX:",round(time_CPLEX,4),"seg.")
        print("Tempo de Execução Relaxação Lagrangeana:",round(time_RL,4),"seg.")
        print("=========================================")
        print("=========================================")
        print("Upper Bound Máximo: ", max(upper_bounds))
        print("Upper Bound Médio: ", np.mean(upper_bounds))
        print("Upper Bound Mínimo: ", min(upper_bounds))
        print("=========================================")
        print("Lower Bound Máximo: ", max(lower_bounds)) 
        print("Lower Bound Médio: ", np.mean(lower_bounds)) 
        print("Lower Bound Mínimo: ", min(lower_bounds)) 
        print("=========================================")
        print("Número de Atualizações do Multiplicador: ", contador_pi) 
        print("=========================================")
        print("#########################################")
 # Salvando Log em arquivo
        filename = "LOG_RESULTADOS_SCP_v2.txt"
        with open(filename, 'a') as file:
            file.write("=========================================\n")
            file.write(f"INSTÂNCIA: {i}\n")
            file.write("=========================================\n")
            file.write(f"ÓTIMO CPLEX: {cplex_optimal}\n")
            file.write(f"RELAXAÇÃO LAGRANGEANA: {max(lower_bounds)}\n")
            file.write(f"PRECISÃO DO MÉTODO: {precisão}%\n")
            file.write("=========================================\n")
            file.write(f"GAP: {GAP}%\n")
            file.write(f"GAP Primal: {GAP_Primal}%\n")
            file.write(f"GAP Dual: {GAP_Dual}%\n")
            file.write("=========================================\n")
            file.write(f"Tempo de Execução CPLEX: {round(time_CPLEX, 4)} seg.\n")
            file.write(f"Tempo de Execução Relaxação Lagrangeana: {round(time_RL, 4)} seg.\n")
            file.write("=========================================\n")
            file.write(f"Upper Bound Máximo: {max(upper_bounds)}\n")
            file.write(f"Upper Bound Médio: {np.mean(upper_bounds)}\n")
            file.write(f"Upper Bound Mínimo: {min(upper_bounds)}\n")
            file.write("=========================================\n")
            file.write(f"Lower Bound Máximo: {max(lower_bounds)}\n")
            file.write(f"Lower Bound Médio: {np.mean(lower_bounds)}\n")
            file.write(f"Lower Bound Mínimo: {min(lower_bounds)}\n")
            file.write("=========================================\n")
            file.write(f"Número de Atualizações do Multiplicador: {contador_pi}\n")
            file.write("=========================================\n")
            file.write("#########################################\n")
        print(f"Log salvo no arquivo {filename}")



        # Geração do gráfico de comparação
        #plot_bounds_comparison(upper_bounds, lower_bounds, cplex_optimal, instancia)
#=================================================================================