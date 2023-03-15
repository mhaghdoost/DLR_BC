# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:31:20 2023

@author: reza_ma
"""
if __name__ == '__main__':
    ##
    import numpy as np
    import matplotlib.pyplot as plt

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter

    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter


    # TODO: fix bug

    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=2,
                             n_obj=2,
                             n_ieq_constr=2,
                             xl=np.array([-2, -2]),
                             xu=np.array([2, 2]))

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = 100 * (x[0] ** 2 + x[1] ** 2)
            f2 = (x[0] - 1) ** 2 + x[1] ** 2

            g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
            g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

            out["F"] = [f1, f2]
            out["G"] = [g1, g2]


    problem = MyProblem()

    algorithm = NSGA2(pop_size=100)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
    algorithm2 = CTAEA(ref_dirs=ref_dirs)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 600),
                   verbose=False,
                   seed=1)

    res2 = minimize(problem,
                    algorithm2,
                    ("n_gen", 200),
                    seed=1,
                    verbose=False)
    ##
    # %%
    # plt.figure()
    sc = Scatter(legend=True, angle=(45, 30))
    sc.addnumbers(problem.pareto_front(ref_dirs), plot_type='surface', alpha=0.2, label="PF", color=["blue"])
    sc.addnumbers(res2.F, facecolor="none", edgecolor="blue", label=res2.algorithm)

    sc.addnumbers(res.F, facecolor="none", edgecolor="black", label=res.algorithm)
    sc.show()
    print("END OF FILE")
