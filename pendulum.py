# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:39:22 2018

@author: Ulysse REGLADE
"""

import sympy as sy
import numpy as np

#The number of consecutives penduliums

class pendulum(object):
    
    def __init__(self, lengths, masses):
        """
        Creat the formal Lagrangian of an n-pandulium.
        lengths conteins the lengths of the differents parts, the length of
        this list is n. masses conteins the masses of the points, its length
        is n+1.
        """
        
        #Parameters of the pendulum
        self.n = len(lengths)
        self.lengths = lengths
        self.masses = masses
        
        #Parameters of physic
        self.t = sy.Symbol('t')
        self.g = -9.81
        
        #Parameters of the pendulum
        self.u = sy.Function('u')(self.t)
        self.x = sy.Function('x')(self.t)
        self.x_1 = sy.diff(self.x, self.t)
        self.x_2 = sy.diff(self.x_1, self.t)
        self.thetas = []
        self.thetas_1 = []
        self.thetas_2 = []
        for i in range(self.n):
            self.thetas.append(sy.Function('theta'+str(i))(self.t))
            self.thetas_1.append(sy.diff(self.thetas[-1], self.t))
            self.thetas_2.append(sy.diff(self.thetas_1[-1], self.t))
        
        #Calculation of the positions of the differents parts of the pendulum
        self.positions = []
        self.positions.append(np.array([self.x, 0]))
        for i in range(self.n):
            position = (self.positions[-1] +
                        np.array([self.lengths[i]*sy.sin(self.thetas[i]),
                                  -self.lengths[i]*sy.cos(self.thetas[i])]))
            self.positions.append(position)
        
        #Calculation of the velocities
        self.quad_velos = [self.x_1**2]
        for i in range(self.n):
            quad_velo = (sy.diff(self.positions[i+1][0], self.t)**2 +
                         sy.diff(self.positions[i+1][1], self.t)**2)
            self.quad_velos.append(quad_velo)
        
        #Calculation of the formal Lagrangian
        print("Computing the Lagrangian....")
        
        self.L = 0.5*self.masses[0]*self.quad_velos[0]
        for i in range(self.n):
            #Kinetic energy
            self.L += 0.5*self.masses[i+1]*self.quad_velos[i+1]
            #Pentential energy
            self.L += self.g*self.masses[i+1]*self.positions[i+1][1]
        #The control
        self.L += self.x*self.u
        
        self.L = sy.simplify(self.L)
        print(self.L)
        
        #Calculation of the dynamic of the system
        self.dynamic = self.formal_dynamic()
        
        #Seting the control matrix to zeros
        self.control = np.zeros((2*(self.n+1)))
        y_ref = [0, 0]
        for i in range(self.n):
            y_ref.append(np.pi)
            y_ref.append(0)
        self.y_ref = np.array(y_ref)
        
        #Lambdification of the functions for optimisation.
        #Lambda functions doesn't support derivatives, so their
        #is some work arround here.
        print("Lambdifyinig the functions.....")
        
        for key in self.dynamic:
            self.dynamic[key] = self.dynamic[key].subs(self.x_1,
                                                       sy.Symbol('x_1'))
            for i in range(self.n):
                self.dynamic[key]=self.dynamic[key].subs(self.thetas_1[i],
                                                         sy.Symbol('theta' +
                                                                   str(i) +
                                                                   '_1'))
        
        self.x_1 = sy.Symbol('x_1')
        for i in range(self.n):
            self.thetas_1[i] = sy.Symbol('theta'+str(i)+'_1')

        sub = [self.x, self.x_1]
        out = [self.dynamic[self.x_2]]
        for i in range(self.n):
            sub.append(self.thetas[i])
            sub.append(self.thetas_1[i])
            out.append(self.dynamic[self.thetas_2[i]])
        sub.append(self.u)
        
        self.lambda_dynamic = sy.lambdify(sub, out)
        self.even_indexs = np.array([2*i for i in range(self.n+1)])
        self.odd_indexs = self.even_indexs +1
        
        print("Dynamic implemented !")
    
    def formal_dynamic(self):
        """
        return the formal dynamicc of the pendulim, the first equation
        correspond to the value of x_2. The others to the values of the
        diffferents theta_2.
        """
        
        print("Creating the formal system (can take few secondes)....")
        left = sy.simplify(sy.diff(sy.diff(self.L, self.x_1), self.t))
        rigth = sy.simplify(sy.diff(self.L, self.x))
        system = [left-rigth]
        val = [self.x_2]
        for i in range(self.n):
            left=sy.simplify(sy.diff(sy.diff(self.L, self.thetas_1[i]),self.t))
            rigth = sy.simplify(sy.diff(self.L, self.thetas[i]))
            system.append(left-rigth)
            val.append(self.thetas_2[i])
            
        print("Solving the system (can take few minutes).....\n")
        solution = sy.solve(system, val)
        solution = sy.simplify(solution)
        
        print("The system is the following:\n")
        print(str(self.x_2)+" = "+str(solution[self.x_2])+"\n")
        for i in range(self.n):
            print(str(self.thetas_2[i])+" = "+
                  str(solution[self.thetas_2[i]])+"\n")
        print("\n")
        
        return solution
    
    def f_from_sympy_exp(self, y, t):
        """
        The actual function to use in the ode integration. takes an y of the
        forme: [x, x_1, theta0, theta0_1, ...]
        NOTE: The excecution of this function is long, due to sympy
        evaluation.
        """
        
        #Creation of the dictionnary that's going to evaluate the expression
        sub = {self.x:y[0], self.x_1:y[1]}
        for i in range(self.n):
            sub.update({self.thetas[i]:y[2*(i+1)],
                        self.thetas_1[i]:y[2*(i+1)+1]})
        
        #Evaluation of the expression
        values = [y[1], self.dynamic[self.x_2].evalf(subs=sub)]
        for i in range(self.n):
            values.append(y[2*(i+1)+1])
            values.append(self.dynamic[self.thetas_2[i]].evalf(subs=sub))

        return values
    
    def f_from_lambda(self, y, t):
        """
        Do the same thing than the function abouve, but it is faster.
        """
        
        #Pendulum interne dynamic
        dydt = np.zeros((2*(self.n+1)))
        dydt[self.even_indexs] = y[self.odd_indexs]
        u = np.sum(self.control*(y-self.y_ref))
        dydt[self.odd_indexs] = self.lambda_dynamic(*y, u)
        
        #Pendulum control
        return dydt