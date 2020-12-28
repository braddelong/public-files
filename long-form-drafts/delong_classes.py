# delong_classes.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class solow:
    
    """ 
    Implements the Solow growth model calculation of the 
    capital-output ratio κ and other model variables
    using the update rule:
    
    κ_{t+1} = κ_t + ( 1 - α) ( s - (n+g+δ)κ_t )
    
    Built upon and modified from Stachurski-Sargeant 
    <https://quantecon.org> class **Solow** 
    <https://lectures.quantecon.org/py/python_oop.html>
    """
    
    def __init__(self, n=0.01,              # population growth rate
                       s=0.20,              # savings rate
                       δ=0.03,              # depreciation rate
                       α=1/3,               # share of capital
                       g=0.01,              # productivity
                       κ=0.2/(.01+.01+.03), # current capital-labor ratio
                       E=1.0,               # current efficiency of labor
                       L=1.0):              # current labor force  

        self.n, self.s, self.δ, self.α, self.g = n, s, δ, α, g
        self.κ, self.E, self.L = κ, E, L
        self.Y = self.κ**(self.α/(1-self.α))*self.E*self.L
        self.K = self.κ * self.Y
        self.y = self.Y/self.L
        self.α1 = 1-((1-np.exp((self.α-1)*(self.n+self.g+self.δ)))/(self.n+self.g+self.δ))
        self.initdata = vars(self).copy()
        
    def calc_next_period_kappa(self):
        "Calculate the next period capital-output ratio."
        # Unpack parameters (get rid of self to simplify notation)
        n, s, δ, α1, g, κ= self.n, self.s, self.δ, self.α1, self.g, self.κ
        # Apply the update rule
        return (κ + (1 - α1)*( s - (n+g+δ)*κ ))

    def calc_next_period_E(self):
        "Calculate the next period efficiency of labor."
        # Unpack parameters (get rid of self to simplify notation)
        E, g = self.E, self.g
        # Apply the update rule
        return (E * np.exp(g))

    def calc_next_period_L(self):
        "Calculate the next period labor force."
        # Unpack parameters (get rid of self to simplify notation)
        n, L = self.n, self.L
        # Apply the update rule
        return (L*np.exp(n))

    def update(self):
        "Update the current state."
        self.κ =  self.calc_next_period_kappa()
        self.E =  self.calc_next_period_E()
        self.L =  self.calc_next_period_L()
        self.Y = self.κ**(self.α/(1-self.α))*self.E*self.L
        self.K = self.κ * self.Y
        self.y = self.Y/self.L

    def steady_state(self):
        "Compute the steady state value of the capital-output ratio."
        # Unpack parameters (get rid of self to simplify notation)
        n, s, δ, g = self.n, self.s, self.δ, self.g
        # Compute and return steady state
        return (s /(n + g + δ))

    def generate_sequence(self, T, var = 'κ', init = True):
        "Generate and return time series of selected variable. Variable is κ by default. Start from t=0 by default."
        path = []
        
        # initialize data 
        if init == True:
            for para in self.initdata:
                 setattr(self, para, self.initdata[para])

        for i in range(T):
            path.append(vars(self)[var])
            self.update()
        return path


class malthusian:
    
    """
    Implements the Malthusian Model with:
    
    1. population growth 
        n =  β*(y/(ϕ ysub)-1)
        
    2. growth of efficiency-of-labor 
        g = h-n/γ
    """
    def __init__(self,
                 L = 1,               # initial labor force
                 E = 1/3,             # initial efficiency of labor
                 K = 3.0,             # initial capital stock
                 
                               # determinants of n (population growth):
                 β = 0.025,           # responsiveness of population growth to increased prosperity.
                 ϕ = 1,               # luxuries parameter
                 ysub = 1,            # subsistence level
                 
                               # determinants of g(efficiency-of-labor growth):
                 h = 0,               # rate at which useful ideas are generated
                 γ = 2.0,             # effect-of-resource scarcity parameter  
                 
                 s = 0.15,            # savings-investment rate
                 α = 0.5,             # orientation-of-growth-toward-capital parameter
                 δ = 0.05,            # deprecation rate on capital parameter
                ):
        self.L, self.E, self.K, self.h, self.γ, self.s, self.α, self.δ = L, E, K, h, γ, s, α, δ 
        self.β, self.ϕ, self.ysub = β, ϕ, ysub
        
        # production (or output)
        self.Y = self.K**self.α*(self.E*self.L)**(1-self.α) 
        self.y = self.Y/self.L
        
        # capital-output ratio 
        self.κ = self.K/self.Y    
        
        # population growth 
        self.n = self.β*((self.y/(self.ϕ*self.ysub)) - 1)
        
        # growth rate of efficiency-of-labor
        self.g = self.h-self.n/self.γ
        
        # store initial data
        self.initdata = vars(self).copy()
    
    def update(self):
        # unpack parameters
        K, s, Y, δ, L, n, E, g, α =self.K, self.s, self.Y, self.δ, self.L, self.n, self.E, self.g, self.α
        β, ϕ, ysub, h, γ = self.β, self.ϕ, self.ysub, self.h, self.γ
        
        #update variables        
        K = s*Y + (1-δ)*K
        L = L*np.exp(n)
        E = E*np.exp(g)
        Y = K**α*(E*L)**(1-α)
        y = Y/L
        κ = K/Y
        n = β*(y/(ϕ*ysub)-1)
        g = h-n/γ
                
        #store variables
        self.K, self.s, self.Y, self.δ, self.L, self.n, self.E, self.g, self.α = K, s, Y, δ, L, n, E, g, α
        self.κ, self.y = κ, y
        
    def gen_seq(self, t, var = 'κ', init = True, log = False):
        "Generate and return time series of selected variable. Variable is κ by default."
        
        path = []
        
        # initialize data 
        if init == True:
            for para in self.initdata:
                 setattr(self, para, self.initdata[para])

        for i in range(t):
            path.append(vars(self)[var])
            self.update()
        
        if log == False:
            return path
        else:
            return np.log(np.asarray(path))

    def steady_state(self, disp = True):
        "Calculate variable values in the steady state"
        #unpack parameters
        s, γ, h, δ, ϕ, ysub, β, α= self.s, self.γ, self.h, self.δ, self.ϕ, self.ysub, self.β, self.α
        
        self.mal_κ = s/(γ*h+δ)
        # malthusian rate of population growth
        self.mal_n = γ*h
        # malthusian standard of living
        self.mal_y = ϕ*(ysub+γ*h/β)
        self.mal_E = self.mal_y*((γ*h+δ)/s)**(α/(1-α))
        
        if display == True:
            return(f'steady-state capital-output ratio κ: {self.mal_κ:.2f}')
            return(f'Malthusian rate of population growth n: {self.mal_n: .2f}')
            return(f'Malthusian standard of living y: {self.mal_y:.2f}')
            return(f'steady-state efficiency-of-labor E: {self.mal_E:.2f}') 
        else: 
            return(self.mal_κ,self.mal_n,self.mal_y,self.mal_E)



class gini:
    """
    For a two-class distribution of income. Initialize 
    the class with a size-of-upper-class variable
    equal to 1/5 and a share-of-upper-class variable
    equal to 4/5
    """
    
    def __init__(self,
                 upper_class = 1/5,       # size of upper class
                 share = 4/5              # share of upper class
                ):
        self.upper_class = upper_class
        self.share = share
        self.gini_value = self.share - self.upper_class
        self.income_ratio = ((self.share/self.upper_class)/
            ((1-self.share)/(1-self.upper_class)))



class 𝜅_convergence_graph:
    
    """
    # This code cell will plot how the the economy's
    # capital-intensity 𝜅 converges to its steady-
    # state balanced-growth level 𝜅^* no matter what
    # the initial condition 𝜅_0.
    #
    # Either accept the values given below for s, n, g, 𝛿,
    # θ, 𝜅_0 and the time T you wish to calculate for,
    # or substitute your own in the relevant code lines:
    """

    def __init__(self, 𝜅_0 = 3,
                       s = 0.20,
                       n = 0.01,
                       g = 0.015,
                       𝛿 = 0.025,
                       θ = 1,
                       T = 200):
        self.𝜅_0, self.s, self.n, self.g, self.𝛿, self.θ, self.T = 𝜅_0, s, n, g, 𝛿, θ, T 

    def draw(self):
        "Draw the convergence graph"
        𝜅_0, s, n, g, 𝛿, θ, T = self.𝜅_0, self.s, self.n, self.g, self.𝛿, self.θ, self.T
        𝜅_star = s/(n+g+𝛿)
        𝜅_max = 2*𝜅_star
        𝜅_min = 0.5

        𝜅_star_series = [𝜅_star]
        𝜅_series = [𝜅_0]

        for t in range(T):
            𝜅_star_series = 𝜅_star_series + [𝜅_star]
            𝜅_series = 𝜅_series + [𝜅_star + (𝜅_series[t-1] - 𝜅_star)*np.exp(-(n+g+𝛿)/(1+θ))]


        𝜅_convergence_df = pd.DataFrame()
        𝜅_convergence_df['steady_state_capital_intensity'] = 𝜅_star_series
        𝜅_convergence_df['capital_intensity'] = 𝜅_series

        ax = plt.gca()
        
        𝜅_convergence_df.capital_intensity.plot(ax=ax)
        𝜅_convergence_df.steady_state_capital_intensity.plot(ax=ax,
                 title = 'Convergence of Capital-Intensity to Steady-State κ*')
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital-Intensity")

