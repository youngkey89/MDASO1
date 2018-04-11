import array
import random
import warnings
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, benchmarks

warnings.filterwarnings("ignore")

# Store material list in dataframe
ml = pd.read_csv("material list.csv", index_col="Material")

x = [None]*5

# Material list for beam and support
Ibeam = ['A1 6061', 'A36 Steel', 'A514 Steel', 'Titanium']
Support = ['A1 6061', 'A36 Steel', 'A514 Steel', 'Concrete']


def main():

    # Dataframe to store the result
    columns = ['no. beam', 'beam material', 'support material', 'optimal cost', 'optimal dimention']
    df = pd.DataFrame(columns=columns)

    for n in range(1, 5):
        for B in Ibeam:
            b = list(ml.loc[B])
            for S in Support:
                s = list(ml.loc[S])
                def Cost(x, b=b, s=s, n=n):

                    # Objective Equation
                    M_Ibeam = (2 * x[0] * x[1] + (x[2] - 2 * x[1]) * x[1]) * 30 * b[0] * n
                    M_support = x[3] * x[4] * 5 * s[0]
                    return b[3] * M_Ibeam + s[3] * M_support,


                def feasible(x, b=b, s=s, n=n):
                    # Constrain Equation
                    M_Ibeam = (2 * x[0] * x[1] + (x[2] - 2 * x[1]) * x[1]) * 30 * b[0] * n
                    I_Ibeam = ((x[2] - 2 * x[1]) ** 3 * x[1] / 12) + 2 * (
                    (x[1] ** 3 * x[0] / 12) + x[1] * x[0] * ((x[2] / 2) + (x[1] / 2)) ** 2)
                    sigma_ibeam = ((7425 * 10 ** 4) + M_Ibeam * 73.575) * (x[2] / 2) / (8 * I_Ibeam * n)
                    tau_Ibeam = (M_Ibeam * 9.81 + 99 * 10 ** 5) / \
                                (4 * (2 * x[0] * x[1] + (x[2] - 2 * x[1])) * n)
                    P_applied = (M_Ibeam * 9.81 + 99 * 10 ** 5) / 2
                    P_Crit = np.pi ** 2 * s[1] * min(x[3] ** 3 * x[4] / 12, x[3] * x[4] ** 3 / 12) / 100
                    sigma_support = P_applied / (x[3] * x[4])

                    g1 = 2 * x[1] / x[2] <= 1
                    g2 = x[1] / x[0] <= 1
                    g3 = sigma_ibeam <= b[2]
                    g4 = tau_Ibeam <= b[2]
                    g5 = P_applied <= P_Crit
                    g6 = sigma_support <= s[2]
                    g7 = 0.1 <= x[0]
                    g8 = 0.01 <= x[1]
                    g9 = 0.1 <= x[2]
                    g10 = 0.2 <= x[3]
                    g11 = 0.3 <= x[4]

                    if g1 * g2 * g3 * g4 * g5 * g6 * g7 * g8 * g9 * g10 * g11 == 1:
                        return True
                    return False

                s = list(ml.loc[S])

                # Type Definition
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)

                toolbox = base.Toolbox()

                # Attribute generator
                toolbox.register("attribute", random.uniform, 0, 2.1)

                # Structure initializers
                toolbox.register("individual", tools.initRepeat,
                                 creator.Individual, toolbox.attribute, len(x))
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)

                # Evaluator
                toolbox.register("evaluate", Cost)
                toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 10 ** 5))

                # Evolution
                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
                toolbox.register("select", tools.selTournament, tournsize=3)

                random.seed(22)

                pop = toolbox.population(n=300)
                hof = tools.HallOfFame(1)
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("avg", np.mean)
                stats.register("std", np.std)
                stats.register("min", np.min)
                stats.register("max", np.max)

                pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.6, ngen=50,
                                               stats=stats, halloffame=hof, verbose=False)
                hof = list(hof)
                opt = hof[0]
                obj_opt = '%.2f' % Cost(opt)
                ro_opt = ['%.3f' % elem for elem in opt]
                result = [n, B, S, obj_opt, ro_opt]
                newdf = pd.DataFrame([result],  columns=columns)
                df = df.append(newdf, ignore_index=True)

                print(result)
    
    return df

if __name__ == "__main__":
    df = main()
    # Store as csv file
    df.to_csv('result.csv', sep='\t', encoding='utf-8', index=None)