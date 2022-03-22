# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:10:08 2022

@author: Chris Wendler
"""

import numpy as np

class Objective():
    def __call__(self, w):
        raise NotImplementedError()
    def grad(self, w):
        raise NotImplementedError()
        
def plot_2d_objective(ax, obj, xrange=[-20,20], yrange=[-5,5], ticks=1000):
    """
    Call with, e.g., ax = plt.axes(projection='3d') as argument.
    """
    # Make data
    x_ticks = np.linspace(xrange[0], xrange[1], ticks)
    y_ticks = np.linspace(yrange[0], yrange[1], ticks)
    x, y = np.meshgrid(x_ticks, y_ticks)
    z = obj(np.concatenate((x[np.newaxis], y[np.newaxis]), axis=0))
    # Plot the surface
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.5)

def gradient_descent(obj, w_init, learning_rate=0.1, n_steps=100000, tol=1e-6, normalize=False):
    w_curr = w_init
    w_hist = [w_init]
    obj_hist = [obj(w_init)]
    for step in range(n_steps):
        direction = obj.grad(w_curr)
        if normalize:
            direction /= np.linalg.norm(direction)
        w_next = w_curr - learning_rate*direction
        if np.abs(obj(w_curr) - obj(w_next)) < tol:
            w_curr = w_next
            w_hist += [w_curr]
            obj_hist += [obj(w_curr)]
            print(f'Steps : {step}')
            break
        w_curr = w_next
        w_hist += [w_curr]
        obj_hist += [obj(w_curr)]
    return w_curr, np.asarray(w_hist), np.asarray(obj_hist)

def gradient_descent_momentum(obj, w_init, learning_rate=0.1, beta=0.9, n_steps=100000, tol=1e-6, normalize=False):
    w_curr = w_init
    z_curr = obj.grad(w_init)
    w_hist = [w_init]
    obj_hist = [obj(w_init)]
    for step in range(n_steps):
        grad = obj.grad(w_curr)
        if normalize:
            grad /= np.linalg.norm(grad)
        z_next = beta*z_curr + grad 
        w_next = w_curr - learning_rate*z_next
        if np.abs(obj(w_curr) - obj(w_next)) < tol:
            w_curr = w_next
            z_curr = z_next
            w_hist += [w_curr]
            obj_hist += [obj(w_curr)]
            print(f'Steps : {step}')
            break
        w_curr = w_next
        z_curr = z_next
        w_hist += [w_curr]
        obj_hist += [obj(w_curr)]
    return w_curr, np.asarray(w_hist), np.asarray(obj_hist)