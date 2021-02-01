# Optimal Prediction using Learning and Shape Optimization

This repository contains code solution for learning-based optimal prediction of distributed parameter systems using sensor shape optimization.

The problem is coded in **Python 3.6.10** and machine learning package **Keras 2.4.0**.

To set the parameters of the problem, use `parametrs.py`. 

Run `shape_optimizar.py` to find an optimal solution. This includes an optimal sensor arangment `omega_b`, solution `u_real`, prediction `u_pred`, cost values in each iteration of the optimization algorithm `cost_values`. The results will be saved in the directory `.\data\` as python npy files.

Run `plotter.py` to plot the results and save them in the path `.\figs\`. Uncomment the subsection: *"to create animation"* in this file in order to generate gif files in the directory `.\gifs\`. 
