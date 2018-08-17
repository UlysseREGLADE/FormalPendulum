import numpy as np

def optimize(f, n, it , center0=None, rnd0 = 1, simplex0 = None):
    """
    Take a cost function f.
    f:R**n -> R
    Applies the simplexe optimizer methode, with it number of iterations.
    If simplex0 is given, the optimisation starts by this simplex.
    Otherwise, generates a simplex around center0 with a randome factor of
    size rnd0 in each dimentions.
    """
    #Initialisation of the simplex
    if(simplex0==None):
        if(center0==None):
            simplex0 = np.zeros((n))
        simplex0=np.array([simplex0+(np.random.rand(n)-0.5)*rnd0
                           for i in range(n+1)])
    simplex = np.zeros((n+1, n+1))
    simplex[:, :-1] = simplex0
    for i in range(n+1):
        simplex[i, -1] = f(simplex[i, :-1])
    simplex = np.array(sorted(simplex, key=lambda x: x[-1]))
    
    stuck_matrix = np.zeros((n+1))
    
    #Optimisation loop
    for i in range(it):
        print("Iteration number: " + str(i))
        print("Best value for this iteration: " + str(simplex[0,-1]))
        print("Best candidate: " + str(simplex[0,:-1]) + "\n")
        
        center = np.average(simplex[:, :-1], 0)
        direction = simplex[-1, :-1] - center
        
        candidates = [simplex[-1, :]]
        for j in range(-8, 8+1):
            if(j!=2 and j!=0):
                new = np.zeros((n+1))
                new[:-1] = center+(j/2)*direction
                new[-1] = f(new[:-1])
                candidates.append(new)
        
        candidates = np.array(sorted(candidates, key=lambda x: x[-1]))
        #print(simplex)
        #print(candidates)
        simplex[-1] = candidates[0]
        simplex = np.array(sorted(simplex, key=lambda x: x[-1]))
    
    return simplex[0]

#def f(x):
#    return (x-3)**2

#optimize(f, 1, 10)