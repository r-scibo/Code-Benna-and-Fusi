#%%
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import math

# First try an implmentation of basic hopfield model

class HopfieldNetwork:
    '''
    Methods
    _____________
    store_memory
        - Inputs: memory to be stored
        - Modiefies synapse matrix to store the memory
        
    overlap
        - Inputs: memory to compute the overlap with
        - Computes the overlap between memory and current status
        
    run_dynamics
        - Inputs: number of iterations, temperature
        - Runs asynchronous dynamics of the network from the current staus
    '''

    # m = 4, levels = 35, alpha=0.25, decreasing_levels=True, beta =2
    def __init__ (self, N, synapse_type = None, syn_specs={}, initial_J = None):
        '''
        Takes as inputs:
            - N: number of neurons
            - initial_J: if desired an initial synaptic configuration
            - synapse_type: type of synapse you want it to have. Possibilities:
                - None: basic hebbian synapse
                - "BfLin": Benna and Fusi 2016 synapse linear 
            - syn_specs: disctionary with the parameters needed for each type of synapse
        '''
        
        self.N = N   
        self.synapse_type = synapse_type
        
        # Current neuronal state
        self.n = np.random.choice([-1, +1], size=N)   # random ±1 initial state   
    
        # Initial synapse matrix
        if synapse_type == None:
            if initial_J == None:
                self.J = np.zeros((N,N), dtype=np.float64)

        if synapse_type == "BfLin":
            if initial_J != None:
                raise Exception("Initial J not implemented for BfLin")
            else:
                self.J_matrix = BfSynapsesVectorized(N, syn_specs["m"], syn_specs["levels"], 
                                                    syn_specs["alpha"], syn_specs["decreasing_levels"], 
                                                    syn_specs["beta"])
                self.J = self.J_matrix.get_J()


            
        # Number of memories stored
        self.p = 0 
        
    def store_memory (self, memory):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")

        delta_j = 0
        if self.synapse_type == None:
            delta_J = basic_hebb(self, memory)
            self.J += np.float64(delta_J)

        if self.synapse_type == "BfLin":
            delta_J = basic_hebb_not_normalized(memory)
            self.J_matrix.update(delta_J)
            self.J = self.J_matrix.get_J()
            
        
        self.p += 1
        print(f"Stored one memory. Number of stored memories = {self.p}")
        return
    
    def overlap(self, memory):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")

        m = (1/self.N)*(self.n @ memory)
        
        return m
        
    def run_async(self, k_sweeps):
        """
        Asynchronous Hopfield updates.
        Each 'sweep' is N attempted flips, chosen at random.
        
        Parameters
        ----------
        k_sweeps : int
            Number of full sweeps (N updates) to perform.
        """
        N = self.N
        state = self.n            # direct view on self.n
        J = self.J.copy()

        # 1) Compute initial fields: h_j = sum_k J[j,k] * state[k]
        h = J.dot(state).astype(np.float64)      # one O(N^2) at start

        # 2) Pre-generate random picks to avoid Python RNG overhead inside the loop
        picks = np.random.randint(N, size=N * k_sweeps)

        # 3) Main loop: for each picked neuron i
        for i in picks:
            # decide its new state
            new_si = 1 if h[i] >= 0 else -1
            old_si = state[i]
            if new_si != old_si:
                # flip and update fields in O(N)
                delta = new_si - old_si       # ±2
                state[i] = new_si
                h += delta * J[:, i]          # vectorized update

        return

    
    def init_at_memory(self, memory, epsilon=0):
        '''
        Initializes the neurons at an initial configuration (memory)
            - memory must be (N,) np array of 1 and -1
            - noise in [0,1]
        '''
        if epsilon == 0:
            self.n = memory.copy()
        else:
            noisy_mem = memory.copy()
            flip_mask = np.random.random((self.N)) < epsilon
            # Flip those bits
            noisy_mem[flip_mask] *= -1
            
            self.n = noisy_mem.copy()
            
        return
            

    
        
           
def basic_hebb(Network, memory):
    '''
    Implements basic hebbian synaptic rule for hopfield model
    '''

    delta = np.outer(memory, memory) / Network.N
    np.fill_diagonal(delta, 0)
    return delta

def basic_hebb_not_normalized(memory):
    '''
    Implements basic hebbian synaptic rule for hopfield model without normalizing by N
    '''

    delta = np.outer(memory, memory)
    np.fill_diagonal(delta, 0)
    return delta


def store_k_memories(Network, k, f=0.5):
    '''
    Stores k random memories in the network
    Saves each memory at row i of a numpy array
    f : float in (0,1], optional
        Fraction of neurons to set active (+1) in the sparse variant.
        Ignored if sparse=False. Default is 0.5.
    '''
    N = Network.N
    memories = np.zeros((k, N))
    
    for i in range(k):
        
        if f == 0.5:
            # classical dense coding: each neuron ±1 with equal probability
            pattern = np.random.choice([-1, +1], size=N)
        
        else:
            # sparse coding: exactly f*N active (+1), rest -1
            if not (0 < f <= 1):
                raise ValueError("f must be in (0, 1] for sparse=True")
            
            pattern = np.full(N, -1, dtype=int)
            dim = int(np.round(f * N))
            # choose k distinct positions to activate
            active_idx = np.random.choice(N, size=dim, replace=False)
            pattern[active_idx] = +1
            
        memories[i] = pattern.copy()
        Network.store_memory(pattern)
    
    return memories

        
class BfSynapsesVectorized:
    def __init__(self, N, m, levels, alpha=0.25, decreasing_levels=True, beta=2):
        self.N = N
        self.m = m
        self.alpha = alpha
        self.beta = beta

        # The 3D synaptic state: shape (N, N, m)
        self.u = np.zeros((N, N, m))

        # Precompute g couplings
        self.g = np.array([beta ** (-2*i + 1) * alpha for i in range(m-1)])

        # Compute levels per internal variable
        self.levels = []
        for i in range(m):
            if decreasing_levels:
                slope = (1 - levels) / (m - 1)
                height = math.ceil(slope * i + levels)
                base = height / 2
                lv = np.arange(-base, base + 1, 1)
            else:
                base = levels / 2
                lv = np.arange(-base, base + 1, 1)
            self.levels.append(lv)
    
    def update(self, delta_J):
        '''
        Fully vectorized update over all synapses (N x N) at once
        delta_J: numpy array of shape (N, N)
        '''
        u_copy = self.u.copy()
        m = self.m

        # Level 0 update (input term + coupling to level 1)
        self.u[:, :, 0] = u_copy[:, :, 0] + delta_J + self.g[0] * (u_copy[:, :, 1] - u_copy[:, :, 0])

        # Internal levels 1 to m-2
        for i in range(1, m - 1):
            self.u[:, :, i] = (u_copy[:, :, i] 
                                + self.g[i-1] * (u_copy[:, :, i-1] - u_copy[:, :, i]) 
                                + self.g[i] * (u_copy[:, :, i+1] - u_copy[:, :, i]))

        # Last level m-1 (only coupling to m-2)
        self.u[:, :, m - 1] = u_copy[:, :, m - 1] + self.g[m-2] * (u_copy[:, :, m-2] - u_copy[:, :, m-1])

        # Stochastic discretization vectorized
        for i in range(m):
            self.u[:, :, i] = self.snap_stochastic_vectorized(self.u[:, :, i], self.levels[i])
    
    def get_J(self):
        return self.u[:, :, 0].copy()

    def snap_stochastic_vectorized(self, u_arr, levels):
        '''
        Fully vectorized stochastic discretization.
        '''
        levels = np.asarray(levels)
        u_arr = np.clip(u_arr, levels[0], levels[-1])
        idx = np.searchsorted(levels, u_arr, side='right')

        lower = levels[np.maximum(idx - 1, 0)]
        upper = levels[np.minimum(idx, len(levels) - 1)]

        d_low = np.abs(u_arr - lower)
        d_high = np.abs(upper - u_arr)
        p_lower = d_high / (d_high + d_low)

        random_draws = np.random.rand(*u_arr.shape)
        snapped = np.where(random_draws < p_lower, lower, upper)

        return snapped

                





        
#%%
# PLAYGROUND
# Plots the evolution of the overlap of a memory in a network with:

import time

start = time.perf_counter()

# Number of neurons
neurons =30000
# Number of evolution runs
runs = 20
# Total memories stored before running
tot_mem_stored = 1
# Memory we want to test the overlap with
test = 0

syn_specs1 = {"m": 4, "levels": 30, "alpha": 0.25, "decreasing_levels": False, "beta": 2}

Network1 = HopfieldNetwork(neurons, synapse_type = "BfLin", syn_specs = syn_specs1)
mems = store_k_memories(Network1, tot_mem_stored)
Network1.init_at_memory(mems[test], 0.0)

overlaps = np.zeros((runs,))
for i in range(runs):
    Network1.run_async(1)
    overlaps[i] = Network1.overlap(mems[test])
    
plt.plot(np.arange(0,runs), overlaps)

# Network1.J_matrix.print_internal_vars()

end = time.perf_counter()

print(f"Elapsed: {end - start:.4f} seconds")



# %%

# Plot from Benna and Fusi hopfield network

def overlap_in_time_plot(cat_forgetting, N, n_mem, time, noise, n_sweeps, synapse_type = None, m = 4, levels = 35, alpha=0.25, decreasing_levels=True, beta =2):
    '''
    Inputs:
        - cat_forgetting: 
            - True: at each t the network is initialized at the last stored memory
            - False: at each t the network is initialized at the first stored memory
        - N: number of neurons of the network you want to work on
        - n_mem: number of runs, to avoid run specific behaviour
        - time: number of memories to store
        - noise: epsilon for corrupted memories
        - n_sweeps: number fo sweeps to perform in running the dynamics
        
    What it does:
    Initializes the network at the last stored memory (t-th memory) and 
    at a noisy version of it [in red], then runs the dynamics and 
    computes the overlap between network status and original memory.
    '''
    
    # Create where to store overlpas
    overlaps_standard = np.zeros((n_mem, time))
    overlaps_noise = np.zeros((n_mem, time))
    
    for j in range(n_mem):
        
        # Create a network
        Net = HopfieldNetwork(N, synapse_type, m, levels, alpha, decreasing_levels, beta =2)
        
        # Initialize where to store memories
        memories = np.zeros((time, Net.N))
        
        # Compute overlpas in time
        for i in range(time):
            
            # Store one memory, time t = i
            memories[i] = store_k_memories(Net, 1)
            
            # Provisional placeholder to test either y = i or 1. 
            # Castophic forgetting or memory lifetime
            y = 0
            if cat_forgetting == True:
                y = i
            else:
                y = 1

            # Calculate overlap for uncorrupted cue
            Net.init_at_memory(memories[y])
            Net.run_async(n_sweeps)
            overlaps_standard[j,i] = Net.overlap(memories[y])
            
            # Calculate overlap for noisy cue
            Net.init_at_memory(memories[y], noise)
            Net.run_async(n_sweeps)
            overlaps_noise[j,i] = Net.overlap(memories[y])
            
    # Plot
    plt.figure(figsize=(8, 5))
    x_vals = np.arange(1, time + 1)
    
    for t in range(time):
        x = x_vals[t]
        # Extract overlaps at this time across runs
        y_std = overlaps_standard[:, t]
        y_noi = overlaps_noise[:, t]
        
        # Sort for a clean vertical line
        y_std_sorted = np.sort(y_std)
        y_noi_sorted = np.sort(y_noi)
        
        # Plot the vertical line connecting standard overlaps at time t
        plt.plot(
            [x] * n_mem, y_std_sorted, color='blue', linestyle='-',
            linewidth=0.8,
            alpha=0.6
        )
        # Plot the blue square markers at the actual (unsorted) points
        plt.plot(
            [x] * n_mem, y_std, marker='s', linestyle='None', color='blue',
            markersize=2.5,
            alpha=0.6
        )
        
        # Plot the vertical line connecting noisy overlaps at time t
        plt.plot([x] * n_mem, y_noi_sorted, color='red', linestyle='-',
            linewidth=0.8,
            alpha=0.6
        )
        # Plot the red square markers at the actual (unsorted) points
        plt.plot([x] * n_mem, y_noi, marker='s', linestyle='None', color='red',
            markersize=2.5,
            alpha=0.6
        )
    
    plt.xlabel("Number of memories")
    plt.ylabel("Overlap")
    plt.ylim([-0.05, 1.05])
    plt.title(f"Overlap vs. memory index  (blue: ε=0, red: ε={noise:.2f})")
    plt.show()
        

overlap_in_time_plot(cat_forgetting = True, N = 100, n_mem = 2, time = 800, noise = 0.25, n_sweeps = 20, 
                     synapse_type = "BfLin", m = 4, levels = 20, alpha=0.25, 
                     decreasing_levels=True, beta =2)


# %%

# %%
