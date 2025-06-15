#%%
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import math
from numba import njit, prange

# Everytime I run (setting up virtual invironment compatible with numba)
# cd ~/path/to/Code-Benna-and-Fusi
# source bf_env/bin/activate


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
                - "Wdecay": Hebbian synapse with decay parameter Jij(t+1) = lambda*Jij(t) + (alpha/N)*epsi*epsj
            - syn_specs: disctionary with the parameters needed for each type of synapse
        '''
        
        self.N = N   
        self.synapse_type = synapse_type
        
        # Current neuronal state
        self.n = np.random.choice([-1, +1], size=N)   # random ±1 initial state   
    
        # Initial synapse matrix
        if synapse_type == None:
            if initial_J == None:
                self.J = np.zeros((N,N), dtype=np.float32)

        if synapse_type == "BfLin":
            if initial_J != None:
                raise Exception("Initial J not implemented for BfLin")
            else:
                self.J_matrix = BfSynapsesNumba(N, syn_specs["bf_m"], syn_specs["bf_levels"], 
                                                    syn_specs["bf_alpha"], syn_specs["bf_decreasing_levels"], 
                                                    syn_specs["bf_beta"], parallel=True)
                self.J = self.J_matrix.get_J()

        if synapse_type == "Wdecay":
            self.hebb_lambda = syn_specs["hebb_lambda"]
            self.hebb_alpha = syn_specs["hebb_alpha"]
            if initial_J != None:
                raise Exception("Initial J not implemented for BfLin")
            else:
                self.J = np.zeros((N,N), dtype=np.float32)


            
        # Number of memories stored
        self.p = 0 
        
    def store_memory (self, memory):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")

        delta_j = 0
        if self.synapse_type == None:
            delta_J = basic_hebb(memory, self.N)
            self.J += np.float64(delta_J)

        if self.synapse_type == "BfLin":
            delta_J = basic_hebb_not_normalized(memory)
            self.J_matrix.update(delta_J)
            self.J = self.J_matrix.get_J()

        if self.synapse_type == "Wdecay":
            delta_J = basic_hebb(memory, self.N, self.hebb_alpha)
            self.J = self.hebb_lambda * self.J + delta_J

            
        
        self.p += 1
        print(f"Stored one memory. Number of stored memories = {self.p}")
        return
    
    def overlap(self, memory):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")

        m = (1/self.N)*(self.n @ memory)
        
        return m
        
    def run_async(self, k_sweeps, binarized = False):
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

        if binarized == True:
            J = np.where(J >= 0, 1, -1)

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
        

    def plot_weight_distribution(self):
        """
        Plot the current distribution of weights.
        
        - For Hebbian and Wdecay: histogram of self.J (excluding diagonal).
        - For BfLin: histograms for each internal variable u[:,:,k].
        """
        if self.synapse_type in [None, "Wdecay"]:
            # Exclude diagonal
            J_no_diag = self.J[~np.eye(self.N, dtype=bool)].flatten()
            
            plt.figure(figsize=(6, 4))
            plt.hist(J_no_diag, color='blue', alpha=0.7)
            plt.title("Synaptic weight distribution")
            plt.xlabel("J_ij")
            plt.ylabel("Count")
            plt.grid(True)
            plt.show()
            
        elif self.synapse_type == "BfLin":
            u = self.J_matrix.u  # shape (N, N, m)
            m = u.shape[2]
            plt.figure(figsize=(12, 3*m))
            for k in range(m):
                plt.subplot(m, 1, k+1)
                u_k = u[:, :, k].flatten()
                plt.hist(u_k, color='green', alpha=0.7)
                plt.title(f"Internal variable u[:,:,{k}]")
                plt.xlabel("Value")
                plt.ylabel("Count")
                plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        else:
            print("Synapse type not recognized for weight distribution plot.")
                

    
        
           
@njit(parallel=False)
def basic_hebb(memory, N, alpha=1.0):
    '''
    Implements basic hebbian synaptic rule for hopfield model
    Allows for a parameter to control new memory significance
    '''

    delta = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            delta[i, j] = (memory[i] * memory[j]) / N
    if alpha != 1.0:
        delta *= alpha
    for i in range(N):
        delta[i, i] = 0
    return delta

def basic_hebb_not_normalized(memory):
    '''
    Implements basic hebbian synaptic rule for hopfield model without normalizing by N
    '''

    delta = np.outer(memory, memory).astype(np.float32)
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
    memories = np.zeros((k, N), dtype=np.float32)
    
    for i in range(k):
        
        if f == 0.5:
            # classical dense coding: each neuron ±1 with equal probability
            pattern = np.random.choice([-1, +1], size=N).astype(np.float32)
        
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

        
class BfSynapsesNumba:
    """
    Optimized Benna & Fusi Synapse model with Numba kernel.
    Can run either in serial or parallel mode.
    """

    def __init__(self, N, m, levels, alpha=0.25, decreasing_levels=True, beta=2, parallel=False):
        self.N = N
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.parallel = parallel  # If True, uses parallel update kernel

        self.u = np.zeros((N, N, m), dtype=np.float32)
        self.g = np.array([beta ** (-2 * i + 1) * alpha for i in range(m-1)], dtype=np.float32)

        # Build levels for each internal variable (heterogeneous)
        self.levels_list = []
        for i in range(m):
            if decreasing_levels:
                slope = (1 - levels) / (m - 1)
                height = math.ceil(slope * i + levels)
                base = height / 2
                lv = np.arange(-base, base + 1, 1).astype(np.float32)
            else:
                base = levels / 2
                lv = np.arange(-base, base + 1, 1).astype(np.float32)
            self.levels_list.append(lv)

        # Pad levels into rectangular array for Numba compatibility
        self.max_levels = max(len(lv) for lv in self.levels_list)
        self.levels_array = np.full((m, self.max_levels), 0.0, dtype=np.float32)
        self.levels_len = np.zeros(m, dtype=np.int32)

        for i in range(m):
            n_lv = len(self.levels_list[i])
            self.levels_array[i, :n_lv] = self.levels_list[i]
            self.levels_len[i] = n_lv

    def update(self, delta_J):
        """
        Update synapses using the precompiled Numba kernel.
        """
        if self.parallel:
            bf_update_parallel(self.u, delta_J, self.g, self.levels_array, self.levels_len)
        else:
            bf_update(self.u, delta_J, self.g, self.levels_array, self.levels_len)

    def get_J(self):
        """
        Extract current effective synaptic matrix (first internal variable).
        """
        return self.u[:, :, 0].copy()

                
@njit
def snap(u, levels_row, n_levels):
    """
    Stochastic snapping of a single internal variable u to its allowed levels.
    """
    # Clip if outside allowed range
    if u <= levels_row[0]:
        return levels_row[0]
    if u >= levels_row[n_levels - 1]:
        return levels_row[n_levels - 1]

    # Search the correct bin
    for idx in range(1, n_levels):
        if u < levels_row[idx]:
            lower = levels_row[idx - 1]
            upper = levels_row[idx]

            # Compute stochastic snapping probability
            d_low = abs(u - lower)
            d_high = abs(upper - u)
            p_lower = d_high / (d_high + d_low)

            return lower if np.random.rand() < p_lower else upper

    # Safety fallback (should not reach here)
    return levels_row[n_levels - 1]


# ===========================================================
# NUMBA-COMPILED SERIAL UPDATE FUNCTION (NO PARALLELIZATION)
# ===========================================================

@njit
def bf_update(u, delta_J, g, levels_array, levels_len):
    """
    Core Benna & Fusi update rule — serial version.
    u: synapse state array, shape (N, N, m)
    delta_J: Hebbian input, shape (N, N)
    g: coupling constants, shape (m-1,)
    levels_array: allowed levels, shape (m, max_levels)
    levels_len: number of levels per internal variable, shape (m,)
    """
    N, _, m = u.shape
    u_copy = u.copy()

    for i in range(N):
        for j in range(N):

            # Update fast variable (k = 0)
            u[i, j, 0] = u_copy[i, j, 0] + delta_J[i, j] + g[0] * (u_copy[i, j, 1] - u_copy[i, j, 0])

            # Update intermediate variables (k = 1 to m-2)
            for k in range(1, m-1):
                u[i, j, k] = (u_copy[i, j, k] 
                              + g[k-1] * (u_copy[i, j, k-1] - u_copy[i, j, k]) 
                              + g[k] * (u_copy[i, j, k+1] - u_copy[i, j, k]))

            # Update slowest variable (k = m-1)
            u[i, j, m-1] = (u_copy[i, j, m-1] 
                            + g[m-2] * (u_copy[i, j, m-2] - u_copy[i, j, m-1]))

            # Stochastic snapping for all m internal variables
            for k in range(m):
                levels_row = levels_array[k, :]
                n_levels = levels_len[k]
                u[i, j, k] = snap(u[i, j, k], levels_row, n_levels)


# ===========================================================
# NUMBA-COMPILED PARALLEL UPDATE FUNCTION (WITH PARALLELIZATION)
# ===========================================================

@njit(parallel=True)
def bf_update_parallel(u, delta_J, g, levels_array, levels_len):
    """
    Same update rule as above, but fully parallelized across synapses.
    """
    N, _, m = u.shape
    u_copy = u.copy()

    for idx in prange(N * N):  # parallel loop across all synapses
        i = idx // N
        j = idx % N

        u[i, j, 0] = u_copy[i, j, 0] + delta_J[i, j] + g[0] * (u_copy[i, j, 1] - u_copy[i, j, 0])

        for k in range(1, m-1):
            u[i, j, k] = (u_copy[i, j, k] 
                          + g[k-1] * (u_copy[i, j, k-1] - u_copy[i, j, k]) 
                          + g[k] * (u_copy[i, j, k+1] - u_copy[i, j, k]))

        u[i, j, m-1] = (u_copy[i, j, m-1] 
                        + g[m-2] * (u_copy[i, j, m-2] - u_copy[i, j, m-1]))

        for k in range(m):
            levels_row = levels_array[k, :]
            n_levels = levels_len[k]
            u[i, j, k] = snap(u[i, j, k], levels_row, n_levels)




        
#%%
# PLAYGROUND
# Plots the evolution of the overlap of a memory in a network with:

import time

start = time.perf_counter()

# Number of neurons
neurons = 1000
# Number of evolution runs
runs = 10
# Total memories stored before running
tot_mem_stored = 1000
# Memory we want to test the overlap with
test = 999

syn_specs1 = {"bf_m": 4, "bf_levels": 30, "bf_alpha": 0.25, "bf_decreasing_levels": True, "bf_beta": 2}
syn_specsW = {"hebb_lambda" : 0.98, "hebb_alpha" : 4}

Network1 = HopfieldNetwork(neurons, synapse_type = "BfLin", syn_specs = syn_specs1)
mems = store_k_memories(Network1, tot_mem_stored)
Network1.init_at_memory(mems[test], 0.0)

overlaps = np.zeros((runs,))
for i in range(runs):
    Network1.run_async(1)
    overlaps[i] = Network1.overlap(mems[test])
    
plt.plot(np.arange(0,runs), overlaps)

Network1.plot_weight_distribution()

end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} seconds")



# %%

# Plot from Benna and Fusi hopfield network

def overlap_in_time_plot(cat_forgetting, synapse_type, N, n_mem, time, noise, n_sweeps, syn_specs, show_plot = True):
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
        - syn_specs: dictionary of parameters for the specific synapse
        - show_plot: True/False
        
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
        Net = HopfieldNetwork(N, synapse_type, syn_specs)
        
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

    if show_plot:      
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

    return overlaps_standard, overlaps_noise
        
syn_specs2 = {"bf_m" : 4, "bf_levels" : 35, "bf_alpha" : 0.25, "bf_decreasing_levels" : True, "bf_beta": 2}
syn_specs3 = {"hebb_lambda" : 0.99, "hebb_alpha" : 4.0}

hold1, hold2 = overlap_in_time_plot(cat_forgetting = True, synapse_type="Wdecay", N = 500, n_mem = 2, time = 500, noise = 0.25, n_sweeps = 10, 
                     syn_specs = syn_specs3, show_plot = True)






# %% # Calculate max memory lifetime

def mem_lifetime_calc(iterations, synapse_type, N, time, noise, n_sweeps, syn_specs):
    '''
    Calculates the average maximum memory lifetime
    Inputs:
        - iterations: number of trials to average over
        - ... Standard inputs for overlap_in_time_plot
    '''

    max_t_standard = np.zeros((iterations,))
    max_t_noise = np.zeros((iterations,))

    for i in range(iterations):
    
        overlaps_standard, overlaps_noise = overlap_in_time_plot(False, synapse_type, N, 1, time, 
                                                                noise, n_sweeps, syn_specs, False)
        
        # Create boolean masks
        mask_standard = (overlaps_standard > 0.99).any(axis=0)
        mask_noise = (overlaps_noise > 0.99).any(axis=0)

        # Get indices where mask is true
        indices_standard = np.where(mask_standard)[0]
        indices_noise = np.where(mask_noise)[0]

        if indices_standard.size > 0:
            max_t_standard[i] = indices_standard.max()

        if indices_noise.size > 0:
            max_t_noise[i] = indices_noise.max()

    return np.mean(max_t_standard), np.mean(max_t_noise)


syn_specs2 = {"bf_m" : 4, "bf_levels" : 45, "bf_alpha" : 0.25, "bf_decreasing_levels" : True, "bf_beta": 2}
syn_specs3 = {"hebb_lambda" : 0.98, "hebb_alpha" : 4.0}

mem_lifetime_calc(10, synapse_type = "BfLin", N = 200, time = 30, noise = 0.25, 
                  n_sweeps = 10, syn_specs = syn_specs2)







# %%  
# Grid search for optimal parameters for Wdecresing synapses, validation metric being max memory lifetime
# In particular, trying to find optimal hebb_lambda and hebb_alpha

def gridsearch_mem_lifetime(param_ranges, iterations, synapse_type, N, time, noise, n_sweeps, syn_specs, plot_heatmap=True):
    '''
    Grid search over synapse parameters for memory lifetime.

    Inputs:
        - param_ranges: dict like {"hebb_lambda": [num_vals, min_val, max_val], ...}
        - iterations: number of trials for averaging
        - synapse_type, N, time, noise, n_sweeps: standard params
        - syn_specs: dictionary template of synapse parameters
        - plot_heatmap: if True, automatically plots heatmap for 1D/2D searches
    '''

    # Generate grid points
    param_names = list(param_ranges.keys())
    grids = []
    for param in param_names:
        n_vals, min_val, max_val = param_ranges[param]
        grid = np.linspace(min_val, max_val, n_vals)
        grids.append(grid)

    # Build full meshgrid for 1D or 2D grid search
    mesh = np.meshgrid(*grids, indexing='ij')

    # Allocate array to store results
    results = np.zeros_like(mesh[0], dtype=np.float32)

    # Iterate over full parameter grid
    it = np.nditer(results, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        # Build syn_specs for current parameter combination
        syn_specs_curr = syn_specs.copy()
        for i, param in enumerate(param_names):
            value = mesh[i][idx]
            syn_specs_curr[param] = value

        # Compute lifetime for this configuration
        avg_std, avg_noi = mem_lifetime_calc(iterations, synapse_type, N, time, noise, n_sweeps, syn_specs_curr)

        # Store result (you can choose avg_std, avg_noi, or a combination)
        it[0] = avg_std

        print(f"Params {syn_specs_curr} --> Lifetime {avg_std:.2f}")
        it.iternext()

    # Plot heatmap if 1D or 2D grid search
    if plot_heatmap:
        if len(param_names) == 1:
            plt.figure(figsize=(6, 4))
            plt.plot(grids[0], results)
            plt.xlabel(param_names[0])
            plt.ylabel("Memory lifetime")
            plt.title("Grid search result")
            plt.grid()
            plt.show()

        elif len(param_names) == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(results, origin='lower', aspect='auto',
                       extent=(grids[1][0], grids[1][-1], grids[0][0], grids[0][-1]),
                       cmap='viridis')
            plt.colorbar(label="Memory lifetime")
            plt.xlabel(param_names[1])
            plt.ylabel(param_names[0])
            plt.title("Grid search result")
            plt.show()

        else:
            print("Heatmap plotting supported only for 1D or 2D searches.")

    return results, grids


param_ranges = {
    "hebb_lambda": [5, 0.95, 0.999],
    "hebb_alpha": [5, 1.0, 10.0]}

results, grids = gridsearch_mem_lifetime(param_ranges, iterations=5,
                                          synapse_type="Wdecay", N=200, time=70, noise=0.25,
                                          n_sweeps=10, syn_specs={})



# %%
