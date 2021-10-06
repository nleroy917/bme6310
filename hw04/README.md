# BME 6310 - Homework 4
## Problem 1
The mosquito population ($N$) over time ($t$) is modeled by the following equation:

$$
\dfrac{dN}{dt} = RN(1-\dfrac{N}{C}) - \dfrac{rN^{2}}{N_{c}^{2} + N^{2}}
$$

With the number of mosquitoes at $t=0$ to be 10,000, and parameter values initially set to:

$$
R=0.55, \\
C=10^{4}, \\
Nc=10^{4}, \\
r=10^{4}, \\
$$
To solve this initial valued problem, we can use the `scipy.integrate` package like so:

```python
# import runge-kutta integrator
from scipy.integrate import solve_ivp

# conditions
t0 = 0 # days
tf = 50 # days
N_0 = 10000 # skeeters

# constants
R = 0.55 # 1/days
C = 1e4
Nc = 1e4
r = 1e4 # skeeters per day

# solve the model
res = solve_ivp(
    dNdt, 
    t_span=[t0, tf], 
    y0=[N_0],
    method="RK45",
    args=(R, C, r, Nc),
    max_step=0.01 # force a smooth graph
)
```

I changed the value of $C$ to be $10^{10}$ and this drastically altered the behavior of our system:

![Mosquito population over time for various values of](figs/mosquito_pop.png)



This makes sense, intuitively... for very large values of $C$, with initially small values of $N$, that our system will be dominated by the first term that acts with first-order kinetics. However, as $N$ becomes larger relative to C, the first term becomes less and less significant and a system that is more aligned with **Michaelis-Menton** kinetics is observed.

## Problem 2
**a.)** From the given system, we can see that there are **three** distinct equations to analyze:

$$
1.)  \dfrac{dX_{1}}{dt} = I - k_{1}X_{1}
$$

$$
2.) \dfrac{dX_{i}}{dt} = 2k_{i-1}X_{i-1} - k_{i}X_{i}
$$

and

$$
3.) \dfrac{dX_{blood}}{dt} = 2k_{N}X_{N} - D
$$

where $D$ is the death rate of blood cells in the blood stream. Let's analyze the steady-state behavior for each distinct equation type:

###  Analyze Eq. 1

$$
\dfrac{dX_{1}}{dt} = 0 = I - k_{1}X_{1}
$$

$$
I = k_{1}X_{1} \Rightarrow X_{1} = \dfrac{I}{k_{1}}
$$

We can see that at steady-state, the number of cell's in **compartment** 1 will be equal to the introduction rate, $I$ over the rate constant for cell-differentiation **out of** compartment 1. We will use this to build up the solution for **Eq. 2**.

### Analyze Eq. 2
Let's analyze the steady state behavior for compartments **2 and 3**

$$
\dfrac{dX_{2}}{dt} = 0 = 2k_{1}X_{1} - k_{2}X_{2}
$$

rearranging, we obtain:

$$
X_{2} = \dfrac{2k_{1}X_{1}}{k_{2}}
$$

and substituting in the derived equation for $X_{1}$:

$$
X_{2} = \dfrac{2k_{1}}{k_{2}}\dfrac{I}{k_{1}}
$$

$$
X_{2} = \dfrac{2I}{k_{2}}
$$

Now lets analyze **compartment 3**:

$$
\dfrac{dX_{3}}{dt} = 0 = 2k_{2}X_{2} - k_{3}X_{3}
$$


rearranging, we obtain:

$$
X_{3} = \dfrac{2k_{2}X_{2}}{k_{3}}
$$

and substituting in the derived equation for $X_{2}$:

$$
X_{3} = \dfrac{2k_{2}}{k_{3}}\dfrac{I}{k_{2}}
$$

$$
X_{3} = \dfrac{4I}{k_{3}}
$$

We now see a distinct pattern emerging. This solution for the steady-state behavior of **all compartments** can be generalized to the following equation:

$$
X_{i_{ss}} = \dfrac{2^{i-1}I}{k_{i}}
$$

**b.)** We can analyze the final equation to determine how many cells, $I$, need to be committed to the erythropoiesis process in order to produce the required 200 billion red blood cells per day. Taking the steady-state behavior of the final equation for **blood**:

$$
\dfrac{dX_{blood}}{dt} = 0 = 2k_{N}X_{N} - D
$$

We can rearrange and substitute in the analytical solution for $X_{N_{ss}}$ to obtain:

$$
2k_{N}\dfrac{2^{N-1}I}{k_{N}} = D
$$

Simplifying this equation we obtain:

$$
\dfrac{D}{I} = 2^{N}
$$

We can see that there is a **critical ratio** necessary for our system in order to obtain the required amount of blood cells. Put another way, if we require $D$ cells each day, then we need to commit $\dfrac{D}{2^{N}}$ cells to the erythropoiesis process in order to produce the required blood cells per day. For our example of **200 billion** required cells and **10** compartments, we can calculate the value o $I$ as such:

$$
I = \dfrac{200\cdot10^{9}}{2^{10}}
$$

which is precisely **195,132,500 cells**.

### Solving this system with Python
To numerically integrate this solution, I coded up the following function in python to represent the system

```python
def cell_system(_, X: list[float], k: list[float], I: float, D: float = 200e9):
    """
    Function to model the erythropoiesis system.
    
    X - array of floats - each representing the amount of cells in compartment i
    k - array of floats - each one corresponding to the ith compartments loss/gen rate
    I - float - number of cells committed to erythropoiesis per day
    P - float - production of blood cells per day 
    """
    # initially check for negatives
    # as we cant have negative amounts
    # of cells
    for i in range(len(X)):
        if X[i] <= 0:
            X[i] = 0
    
    # init array to return
    dXdt = np.zeros(len(X))
    
    # set value of first equation dX1/dt
    dXdt[0] = I - k[0]*X[0]
    
    # iterate through the next N-1 equations
    for i in range(1,len(X)-1):
        dXdt[i] = 2*k[i-1]*X[i-1] - k[i]*X[i]
    
    # run the final equation for blood
    # by leveraging the i value in the
    # for loop
    i +=1
    dXdt[i] = 2*k[i-1]*X[i-1] - D
    
    return dXdt
```

There are **3 distinct blocks** to this system:

1. Compartment 1
2. Compartments 2 through N
3. The blood system


