#%%
import numpy as np
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
        
    run_async
        - Inputs: number of iterations, temperature
        - Runs asynchronous dynamics of the network from the current staus
    '''

    def __init__ (self, N, synapse_type = None, syn_specs={}, c = None):
        '''
        Takes as inputs:
            - N: number of neurons
            - initial_J: if desired an initial synaptic configuration
            - synapse_type: type of synapse you want it to have. Possibilities:
                - None: basic hebbian synapse
                - "BfLin": Benna and Fusi 2016 synapse linear 
                - "Wdecay": Hebbian synapse with decay parameter Jij(t+1) = lambda*Jij(t) + (alpha/N)*epsi*epsj
            - syn_specs: disctionary with the parameters needed for each type of synapse
            - c: if present specifies the level of sparsity in synaptic connection -> Erdos Renyi network with cN edges
        '''
        
        self.N = N   
        self.synapse_type = synapse_type
        self.c = c
        self.mask = None

        # If sparsely connected network create a mask
        if c != None:
            m = np.random.rand(N, N) < c
            m = np.triu(m, 1)
            self.mask = (m + m.T).astype(int)
            np.fill_diagonal(self.mask, 0)
        else:
            self.mask = np.ones((N, N), dtype=np.int8)
            
        # Current neuronal state
        self.n = np.random.choice([-1, +1], size=N)   # random ±1 initial state   
    
        # Initial synapse matrix
        if synapse_type == None:
            self.J = np.zeros((N,N), dtype=np.float32)

        if synapse_type == "BfLin":
            self.J_matrix = BfSynapsesNumba(N, c, self.mask, syn_specs["bf_m"], syn_specs["bf_levels"], 
                                                    syn_specs["bf_alpha"], syn_specs["bf_decreasing_levels"], 
                                                    syn_specs["bf_beta"])
            self.J = self.J_matrix.get_J()

        if synapse_type == "Wdecay":
            self.hebb_lambda = syn_specs["hebb_lambda"]
            self.hebb_alpha = syn_specs["hebb_alpha"]
            self.J = np.zeros((N,N), dtype=np.float32)


        # Store synapse specifications as attribute of the network
        self.syn_specs = syn_specs
            
        # Number of memories stored
        self.p = 0 


        
    def store_memory (self, memory, f=None):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")

        # Different cases depending on synapse type

        if self.synapse_type == None:

            # Calculate standard hebbian rule
            delta_J = basic_hebb(memory, self.N)

            # If sparsely connected, mask the delta
            if self.c != None:
                delta_J *= self.mask/self.c

            #Update
            self.J += np.float32(delta_J)


        if self.synapse_type == "BfLin":

            # Calculate standard hebbian update, but without normalization by N
            delta_J = basic_hebb_not_normalized(memory)

            # Use specific update algorithm from BfSynapse holder object
            self.J_matrix.update(delta_J)

            # Extract the efficacies corresponding to u_1(t)
            self.J = self.J_matrix.get_J()

            # If sparsely connected, mask the delta
            if self.c != None:
                delta_J *= self.mask


        if self.synapse_type == "Wdecay":

            # Calculate hebbian update multiplied by parameter alpha
            delta_J = basic_hebb(memory, self.N, self.hebb_alpha)

            # Mask if sparsely connected
            if self.c != None:
                delta_J *= self.mask/self.c

            # Specific update rule of Weight Decaying synapses
            self.J = self.hebb_lambda * self.J + delta_J


        self.p += 1
        print(f"Stored one memory. Number of stored memories = {self.p}")
        return
    
    def overlap(self, memory):
        """
        Returns an overlap in [-1,1].
        memory, self.n: both ±1 arrays
        f: coding level (fraction of 1's in the 0/1 version)
        """
        if memory.shape != (self.N,):
            raise ValueError("Memory must be of shape (N,) with ±1 entries")


        m = (1/self.N)*(self.n @ memory)

        return m


    # Mess in this function: to add case for binarized runnning and for sparse coding
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
                

        
           

def basic_hebb(memory, N, alpha=1.0, f = 0.5):
    '''
    Implements basic hebbian synaptic rule for hopfield model
    Parameters:
        - alpha to control new memory significance
        - f for the coding level
    '''
    delta = np.outer(memory, memory).astype(np.float32) / N
    np.fill_diagonal(delta, 0)
    if alpha != 1.0:
        delta *= alpha

    return delta

def basic_hebb_not_normalized(memory):
    '''
    Implements basic hebbian synaptic rule for hopfield model without normalizing by N
    '''

    delta = np.outer(memory, memory).astype(np.float32)
    np.fill_diagonal(delta, 0)
    return delta


def store_k_memories(Network, k, f=None):
    '''
    Stores k random memories in the network
    Saves each memory at row i of a numpy array
    Inputs:
        - f : Fraction of neurons to set active (+1) in the sparse variant.
              Initialized to none so the function can also be called for HopfieldNetworks
    '''
    N = Network.N
    memories = np.zeros((k, N), dtype=np.float32)
    
    for i in range(k):

        pattern = 0
        
        # Standard coding
        if f == None:
            # classical dense coding: each neuron ±1 with equal probability
            pattern = np.random.choice([-1, +1], size=N).astype(np.float32)

        # Sparse coding: exactly f*N active (+1), rest 0
        else:
            pattern = np.full(N, 0, dtype=np.float32)

            # Activate f*N neurons
            dim = int(np.round(f * N))

            # Choose indexes to activate and activate them
            active_idx = np.random.choice(N, size=dim, replace=False)
            pattern[active_idx] = +1
            
        # Store memory generated in memory array
        memories[i] = pattern.copy()

        # Store the memory in the network
        Network.store_memory(pattern, f)
    
    return memories

        
class BfSynapsesNumba:
    """
    Optimized Benna & Fusi Synapse model with Numba kernel.
    Can run either in serial or parallel mode.
    """

    def __init__(self, N, c, mask,  m, levels, alpha=0.25, decreasing_levels=True, beta=2):
        self.N = N
        self.c = c
        self.mask = mask
        self.m = m
        self.alpha = alpha
        self.beta = beta

        self.u = np.zeros((N, N, m), dtype=np.float32)
        self.g = np.zeros((m,m), dtype=np.float32)

        # Compute g values such that delta-u_k = g_{k,k-1}*alpha*(u_{k-1} - u_k) + g_{k,k+1}*alpha*(u_{k+1} - u_k)
        # Where g_{k,k-1} = beta^(-2k + 2)   |   g_{k,k+1} = beta^(-2k + 1)   
        # Note that in the mathematical formula i = 1,...,m . We'll have to deal with arrays going from 0,...,(m-1)
        # We will store g_{k,k-1} in the matrix self.g at position (k-1, k-2), and g_{k,k-1} at (k-1, k)

        for i in range(m):
            for j in range(m):
                if j == i-1:
                    self.g[i,j] = beta**(-2*(i+1) + 2)*alpha
                if j == i+1:
                    self.g[i,j] = beta**(-2*(i+1) + 1)*alpha

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
        bf_update_parallel(self.u, self.mask, delta_J, self.g, self.levels_array, self.levels_len)

    def get_J(self):
        """
        Extract current effective synaptic matrix (first internal variable).
        """

        J = self.u[:, :, 0].copy()
        return J


# Numba function to stochastically discretize variables for Bf synapses         
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




# Function to update the Bf synapse matrix. Leverages Numba faster compilation and parallilazion to speed up the process
@njit(parallel=True)
def bf_update_parallel(u, mask, delta_J, g, levels_array, levels_len):
    """
    Core Benna & Fusi update rule — serial version.
    u: synapse state array, shape (N, N, m)
    delta_J: Hebbian input, shape (N, N)
    g: coupling constants, shape (m,m)
    levels_array: allowed levels, shape (m, max_levels)
    levels_len: number of levels per internal variable, shape (m,)
    Same update rule as above, but fully parallelized across synapses.
    """
    N, _, m = u.shape
    u_copy = u.copy()

    for idx in prange(N * N):  # parallel loop across all synapses
        i = idx // N
        j = idx % N

        # If synapse does not exist, skip
        if mask[i, j] == 0:
           continue

        # Otherwise, update the first variable, which receives the hebbian input
        u[i, j, 0] = u_copy[i, j, 0] + delta_J[i, j] + g[1,0] * (u_copy[i, j, 1] - u_copy[i, j, 0])
        
        # Update internal varibles according to formula u_k(t+1) = u_k(t) + alpha*g_{k,k-1}*(u_{k-1}(t) - u_k(t)) + alpha*g_{k,k+1}*(u_{k+1}(t) - u_k(t))
        for k in range(1, m-1):
            u[i, j, k] = (u_copy[i, j, k]
                          + g[k, k-1] * (u_copy[i, j, k-1] - u_copy[i, j, k])
                          + g[k, k+1] * (u_copy[i, j, k+1] - u_copy[i, j, k]))
            
        # Update last variable, which has leakage term = 0 
        u[i, j, m-1] = (u_copy[i, j, m-1]
                        + g[m-1, m-2] * (u_copy[i, j, m-2] - u_copy[i, j, m-1]))

        # Sotchastically discretize the synapses
        for k in range(m):
            val = snap(u[i, j, k], levels_array[k], levels_len[k])
            u[i, j, k] = val  



class SparseCodingNetwork(HopfieldNetwork):
    '''
    Similar implementation to HopfieldNetwork allowing for sparse coding.
    The model will have binary neurons n = 0,1.
    Methods and inputs are named similarly, but are adapted for sparse coding scenario
    '''

    def __init__ (self, N, f, synapse_type = None, syn_specs={}, initial_J = None, c = None):
        '''
        Same intputs as before, with the addition of:
            - f: coding level of memrories (how many neurons are active in each memory)
            - synapse_type: In addition to other types, also supports
                - "DoubleW": double well synapses from Feng & Brunel (2024)
        '''
        
        self.N = N   
        self.f = f
        self.synapse_type = synapse_type

        # Parameters for sparse connection
        self.c = c
        self.mask = None

        # If sparsely connected network create a mask
        if c != None:
            # We generate the mask, each c_ij will be 1 with probability c
            self.mask = (np.random.rand(N, N) < c).astype(int)

            # Ensure elements on diagonal are 0 (no self connections)
            np.fill_diagonal(self.mask, 0)

        else:
            # Generate an all 1 mask
            self.mask = np.ones((N, N), dtype=np.int8)
            
        # Current neuronal state
        self.n = np.random.choice([0, +1], size=N)   # random 0,1 initial state   

        # Initial synapse matrix
        if synapse_type == None:
            self.J = np.zeros((N,N), dtype=np.float32)

        if synapse_type == "BfLin":
            self.J_matrix = BfSynapsesNumba(N, c, self.mask, syn_specs["bf_m"], syn_specs["bf_levels"], 
                                                    syn_specs["bf_alpha"], syn_specs["bf_decreasing_levels"], 
                                                    syn_specs["bf_beta"])
            self.J = self.J_matrix.get_J()

        if synapse_type == "Wdecay":
            self.hebb_lambda = syn_specs["hebb_lambda"]
            self.hebb_alpha = syn_specs["hebb_alpha"]
            self.J = np.zeros((N,N), dtype=np.float32)

        if synapse_type == "DoubleW":
            self.J = np.zeros((N,N), dtype=np.float32)

        # Store synapse specifications as attribute of the network
        self.syn_specs = syn_specs
            
        # Number of memories stored
        self.p = 0 



    def store_memory (self, memory, f=0.5):
        if memory.shape != (self.N,):
            raise Exception("Memory is not in correct shape or format")
        

        if self.synapse_type == None:
            if f == 0.5:
                delta_J = basic_hebb(memory*2 -1, self.N)
                if self.c != None:
                    delta_J *= self.mask/self.c 
                self.J += np.float32(delta_J)
            else:
                delta_J = tsodyks_feigelman(memory, self.f) / (self.N*self.f*(1-self.f))
                if self.c != None:
                    delta_J *= self.mask/self.c 
                self.J += np.float32(delta_J)  

             

        if self.synapse_type == "BfLin":
            if f == 0.5:
                delta_J = basic_hebb_not_normalized(memory*2 -1)
                self.J_matrix.update(delta_J)
                self.J = self.J_matrix.get_J()
            else:
                delta_J = tsodyks_feigelman(memory, self.f)
                self.J_matrix.update(delta_J)
                self.J = self.J_matrix.get_J()


        if self.synapse_type == "Wdecay":
            if f == 0.5:
                delta_J = basic_hebb(memory*2 -1, self.N, self.hebb_alpha)
                if self.c != None:
                    delta_J *= self.mask/self.c 
                self.J = self.hebb_lambda * self.J + delta_J
            else:
                delta_J = tsodyks_feigelman(memory, self.f) * self.hebb_alpha
                if self.c != None:
                    delta_J *= self.mask/self.c 
                self.J = self.hebb_lambda * self.J + delta_J


        if self.synapse_type == "DoubleW":
            # 1) Build the Hebb‐like increment
            if f == 0.5:
                # memory already ±1
                delta_J = basic_hebb(memory*2 -1, self.N)
            else:
                # memory binary 0/1
                delta_J = tsodyks_feigelman(memory, self.f) 

            # 2) Drift + Hebb + (masked) noise
            r1, r2, r3, C = (self.syn_specs[k] for k in ("DW_r1","DW_r2","DW_r3","DW_C"))
            noise = (r3 * np.random.randn(self.N, self.N) * (self.mask if r3 else 0))

            update = -r1 * dU(C, self.J) + r2 * delta_J + noise

            # 3) Mask & normalize
            mask_norm = self.mask.astype(np.float32) / self.mask.mean()
            self.J += update * mask_norm

            # 4) Zero self‐connections
            np.fill_diagonal(self.J, 0)

        np.fill_diagonal(self.J, 0)

        self.p += 1
        print(f"Stored one memory. Number of stored memories = {self.p}")
        return
    
    
    def overlap(self, memory, f=0.5):
        """
        Returns an overlap in [-1,1].
        memory, self.n: both 0,1 arrays
        f: coding level (fraction of 1's in the 0/1 version)
        """
        if memory.shape != (self.N,):
            raise ValueError("Memory must be of shape (N,) with ±1 entries")

        centered = memory - f
        m = (1/(self.N*f*(1-f)))*(self.n @ centered)

        return m
    

    def run_async(self, k_sweeps, binarized = False):

        # If we have Double Well synapses we run parallel update, as written in orginial Feng-Brunel 2024 paper.
        # It goes according to the follwoing update rule: h_i = 1/N * sum[over j](J_{ij} * n_j(t))  n_i(t+1) = Heavyside(h_i - theta)

        if self.synapse_type == "DoubleW":
            for _ in range(k_sweeps):
                # # Center the state so threshold = 0 automatically
                centered_state = self.n - self.f       # now ∈ {–f, 1–f}

                h = self.J.dot(centered_state) / self.N

                # zero threshold on centered field
                self.n = (h >= 0).astype(int)

        else:
                N = self.N
                state = self.n              # direct view of the 0/1 state
                J = self.J                  # already masked & centered

                # 1) Initial fields: h_j = (1/N) sum_k J[j,k] * state[k]
                #    If you'd like to keep the 1/N factor, add it here and in the delta‐update below.
                centered_state = self.n - self.f
                h = J.dot(centered_state).astype(np.float64)

                # 2) Pre‐draw all the random neuron indices
                picks = np.random.randint(0, N, size=N * k_sweeps)

                # 3) Asynchronous updates
                for i in picks:
                    old_si = state[i]
                    new_si = 1 if h[i] >= 0 else 0

                    if new_si != old_si:
                        delta = new_si - old_si      # now delta∈{+1, -1}
                        state[i] = new_si
                        h += delta * J[:, i] 

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
            # Flip those bits: 
            noisy_mem[flip_mask] = 0**noisy_mem[flip_mask]  # as 0**0 = 1 and 0**1 = 0
            self.n = noisy_mem.copy()
            
        return

    

# Calculates the derivative of the potential for double well synaptic model
def dU(C, x):
    '''
    Inputs:
        - C: width of potential well
        - x: (N,N) np array of synaptic efficacies
    '''
    return 2*x + 2*C * np.sign(x)

        
# Calculates synaptic update based on tsodyks feigelman model
def tsodyks_feigelman(memory, f):

    centered = memory - f
    delta = np.outer(centered, centered).astype(np.float32)
    np.fill_diagonal(delta, 0)

    return delta



        
#%%
# PLAYGROUND  - CODING LEVEL STANDARD
# Plots the evolution of the overlap of a memory in a network with:

import time

start = time.perf_counter()

# Number of neurons
neurons = 30000
# Number of evolution runs
runs = 10
# Total memories stored before running
tot_mem_stored = 1
# Memory we want to test the overlap with
test = 0
# Coding level of memories
f = None


syn_specs1 = {"bf_m": 4, "bf_levels": 30, "bf_alpha": 0.25, "bf_decreasing_levels": False, "bf_beta": 2}
syn_specsW = {"hebb_lambda" : 0.98, "hebb_alpha" : 4}

Network1 = HopfieldNetwork(neurons, synapse_type = None, syn_specs = syn_specsW, c = 0.01)
mems = store_k_memories(Network1, tot_mem_stored, f)
Network1.init_at_memory(mems[test], 0.0)

overlaps = np.zeros((runs,))
for i in range(runs):
    Network1.run_async(1, binarized = True)
    overlaps[i] = Network1.overlap(mems[test])
    
plt.plot(np.arange(0,runs), overlaps)

Network1.plot_weight_distribution()

end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} seconds")


#%%

# PLAYGROUND  - SPARSE CODING REGIME
# Plots the evolution of the overlap of a memory in a network with:

import time

start = time.perf_counter()

# Number of neurons
neurons = 1500
# Number of evolution runs
runs = 10
# Total memories stored before running
tot_mem_stored = 50
# Memory we want to test the overlap with
test = 49
# Coding level of memories
f = 0.5


syn_specs1 = {"bf_m": 4, "bf_levels": 30, "bf_alpha": 0.25, "bf_decreasing_levels": True, "bf_beta": 2}
syn_specsW = {"hebb_lambda" : 0.98, "hebb_alpha" : 4}
syn_specsDW = {"DW_r1" : 0.01, "DW_r2" : 1.0, "DW_r3": 0, "DW_C": 1}

Network1 = SparseCodingNetwork(neurons, f, synapse_type = None, syn_specs = syn_specs1, c = None)
mems = store_k_memories(Network1, tot_mem_stored, f)
Network1.init_at_memory(mems[test], 0.0)

overlaps = np.zeros((runs,))
for i in range(runs):
    Network1.run_async(1)
    overlaps[i] = Network1.overlap(mems[test], f)
    
plt.plot(np.arange(0,runs), overlaps)

Network1.plot_weight_distribution()

end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} seconds")



# %%

def overlap_in_time_plot(
    cat_forgetting,
    synapse_type,
    N,
    c,
    n_mem,
    time,
    noise,
    n_sweeps,
    syn_specs,
    binarized,
    f=None,
    show_plot=True
):
    """
    cat_forgetting: if True, test on the most‐recent (t-th) memory; else on the first memory.
    synapse_type: "None", "BfLin", "Wdecay", or "DoubleW"
    N, c: size and connectivity
    n_mem: how many independent runs
    time: how many memories to store per run
    noise: epsilon for corrupted cue
    n_sweeps: sweeps per test
    syn_specs: dict of parameters for this synapse
    binarized: whether to pass binarized=True to run_async
    f: if None → dense regime; if float→ use sparse coding (0<f<1)
    """
    overlaps_standard = np.zeros((n_mem, time))
    overlaps_noise    = np.zeros((n_mem, time))

    for run_idx in range(n_mem):
        # pick network class + coding level
        if f is None and synapse_type != "DoubleW":
            # dense HopfieldNetwork for None, BfLin, Wdecay
            Net = HopfieldNetwork(
                N,
                synapse_type=(None if synapse_type=="None" else synapse_type),
                syn_specs=syn_specs,
                c=c
            )
            f_local = None
        else:
            # sparse coding (either DoubleW or any type when f given)
            f_local = f if (f is not None) else 0.5
            Net = SparseCodingNetwork(
                N,
                f_local,
                synapse_type=(None if synapse_type=="None" else synapse_type),
                syn_specs=syn_specs,
                c=c
            )

        if synapse_type == "DoubleW" and ((f == None) or (f==0.5)):
            store_k_memories(Net, 1, 0.5)
        # we will need to remember the very first memory if cat_forgetting==False
        first_memory = None

        for t in range(time):
            # generate exactly one new pattern
            if f_local is None:
                patt = np.random.choice([-1,1], size=N).astype(np.float32)
            else:
                patt = np.zeros(N, dtype=np.float32)
                k = int(round(f_local * N))
                idx = np.random.choice(N, size=k, replace=False)
                patt[idx] = 1

            # store it
            Net.store_memory(patt, f_local)

            # keep the first memory around
            if t == 0:
                first_memory = patt.copy()

            # choose which memory to cue
            if cat_forgetting:
                cue = patt           # most‐recent
            else:
                cue = first_memory   # always the very first

            # clean test
            Net.init_at_memory(cue, epsilon=0.0)
            Net.run_async(n_sweeps, binarized=binarized)
            if f_local is None:
                overlaps_standard[run_idx,t] = Net.overlap(cue)
            else:
                overlaps_standard[run_idx,t] = Net.overlap(cue, f_local)

            # noisy test
            Net.init_at_memory(cue, epsilon=noise)
            Net.run_async(n_sweeps, binarized=binarized)
            if f_local is None:
                overlaps_noise[run_idx,t] = Net.overlap(cue)
            else:
                overlaps_noise[run_idx,t] = Net.overlap(cue, f_local)

    if show_plot:
        plt.figure(figsize=(8,5))
        x = np.arange(1, time+1)
        for t in range(time):
            y_std = overlaps_standard[:,t]
            y_noi = overlaps_noise[:,t]
            # plot sorted vertical lines + raw points
            plt.plot([t+1]*n_mem, np.sort(y_std), '-', color='blue', alpha=0.6, lw=0.8)
            plt.plot([t+1]*n_mem, y_std, 's', color='blue', alpha=0.6, ms=2.5)
            plt.plot([t+1]*n_mem, np.sort(y_noi), '-', color='red',  alpha=0.6, lw=0.8)
            plt.plot([t+1]*n_mem, y_noi, 's', color='red', alpha=0.6, ms=2.5)

        plt.xlabel("Stored‐memory index t")
        plt.ylabel("Overlap")
        plt.ylim(-0.05,1.05)
        plt.title(f"Overlap of last stored memory vs t  - Weight Decaying synapses")
        plt.show()

    return overlaps_standard, overlaps_noise




syn_specs2 = {"bf_m" : 4, "bf_levels" : 35, "bf_alpha" : 0.25, "bf_decreasing_levels" : True, "bf_beta": 2}
syn_specs3 = {"hebb_lambda" : 0.995, "hebb_alpha" : 4.0}
c = None

# Ns = [100, 200, 400, 800, 1600, 3500, 7000]

#%%
syn_specs3 = {"hebb_lambda" : 0.995, "hebb_alpha" : 4.0}

# dense regime DoubleW
hold1,hold2 = overlap_in_time_plot(
    cat_forgetting=False,
    synapse_type=None,
    N=800,
    c=None,
    n_mem=2,
    time=350,
    noise=0.25,
    n_sweeps=10,
    syn_specs=syn_specs3,
    binarized=False,
    f=None,           # uses f=0.5 internally for DoubleW
    show_plot=True
)

#%%

syn_specsDW =  {"DW_r1":0.01, "DW_r2":1.0, "DW_r3":0, "DW_C":1.2}

# sparse regime 
hold3,hold4 = overlap_in_time_plot(
    cat_forgetting=True,
    synapse_type="Wdecay",
    N=800,
    c=None,
    n_mem=1,
    time=2000,
    noise=0.25,
    n_sweeps=10,
    syn_specs=syn_specs3,
    binarized=False,
    f=None,            # uses SparseCodingNetwork with f=0.1
    show_plot=True
)


# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_max_memory_lifetime(
    Ns,
    synapse_specs,
    max_t_dict,
    threshold,
    runs=10,
    iters_per_store=1,
    coding="dense",
    f=0.5,
):

    """
    For each synapse model and each N in `Ns`, store up to max_t_dict[model][i] memories,
    measuring after each store whether the *first* memory is still retrievable above `threshold`.
    Repeat `runs` times and average the “last t above threshold” as the lifetime.
    Plots lifetime vs N on a log‐x axis for all models, with colored points + interpolated curves.
    
    Parameters
    ----------
    Ns : list of int
        Network sizes, in increasing order.
    synapse_specs : dict
        e.g. {
            "None": {},
            "BfLin": {"bf_m":4, "bf_levels":30, ...},
            "Wdecay": {"hebb_lambda":0.98, "hebb_alpha":4},
            "DoubleW": {"DW_r1":0.01, ...},
        }
    max_t_dict : dict of lists
        e.g. {
          "None":    [100,200,400,800],
          "BfLin":   [ 50,100,200,400],
          "Wdecay":  [200,400,800,1600],
          "DoubleW": [ 50,100,200,400],
        }
        Each list must have len == len(Ns).
    threshold : float
        Overlap threshold for deeming a memory still retrievable.
    runs : int
        Number of independent repeats to average over.
    iters_per_store : int
        How many asynchronous‐sweep iterations to run after each new memory is stored.
    coding : {"dense","sparse"}
        If "dense", uses HopfieldNetwork for all except DoubleW (which uses SparseCodingNetwork with f=0.5).
        If "sparse", uses SparseCodingNetwork(N,f,...) for all models.
    f : float
        Coding level, only used if coding=="sparse".
    """

    colors = {
        "None":    "C0",
        "BfLin":   "C1",
        "Wdecay":  "C2",
        "DoubleW": "C3",
    }

    display_names = {
        "None":    "Unbounded Continuous",
        "BfLin":   "Bidirectional Cascade",
        "Wdecay":  "Weight Decaying",
        "DoubleW": "Double Well",
    }

    fig, ax = plt.subplots(figsize=(8,6))
    ax.grid(True, linestyle='--', linewidth=0.7, color='lightgray', alpha=0.6)

    for model, specs in synapse_specs.items():
        # collect lifetimes for each N and run
        lifetimes = np.zeros((len(Ns), runs), dtype=float)

        for i, N in enumerate(Ns):
            t_max = max_t_dict[model][i]

            for run in range(runs):
                # instantiate network
                if coding=="dense":
                    if model=="DoubleW":
                        net = SparseCodingNetwork(N, 0.5,
                                                  synapse_type="DoubleW",
                                                  syn_specs=specs[N])
                        f_local = 0.5
                    else:
                        net = HopfieldNetwork(N,
                                              synapse_type=(None if model=="None" else model),
                                              syn_specs=specs)
                        f_local = None
                else:  # sparse coding
                    net = SparseCodingNetwork(N, f,
                                              synapse_type=(None if model=="None" else model),
                                              syn_specs=specs)
                    f_local = f

                # pre-generate memories
                memories = []
                for _ in range(t_max):
                    if f_local is None:
                        patt = np.random.choice([-1,1], size=N).astype(np.float32)
                    else:
                        patt = np.zeros(N, dtype=np.float32)
                        idx = np.random.choice(N, size=int(round(f_local*N)), replace=False)
                        patt[idx] = 1.0
                    memories.append(patt)

                # store & test
                overlap_history = np.zeros(t_max, dtype=float)
                for t in range(t_max):
                    net.store_memory(memories[t], f_local)
                    net.init_at_memory(memories[0], 0.0)
                    net.run_async(iters_per_store)
                    overlap_history[t] = net.overlap(memories[0], f_local) \
                        if f_local is not None else net.overlap(memories[0])

                # record lifetime
                above = np.where(overlap_history > threshold)[0]
                lifetimes[i, run] = (above[-1] + 1) if len(above) > 0 else 0

        # compute mean & std dev
        mean_life = lifetimes.mean(axis=1)
        std_life = lifetimes.std(axis=1, ddof=0)

        # plot with legend using display_names
        label = display_names.get(model, model)
        color = colors[model]
        ax.scatter(Ns, mean_life, color=color, label=label, zorder=3)
        N_plot    = np.logspace(np.log10(Ns[0]), np.log10(Ns[-1]), 200)
        mean_plot = np.interp(N_plot, Ns, mean_life)
        std_plot  = np.interp(N_plot, Ns, std_life)
        ax.fill_between(N_plot,
                        mean_plot - std_plot,
                        mean_plot + std_plot,
                        color=color, alpha=0.3, zorder=2)
        ax.plot(N_plot, mean_plot, color=color, lw=1.5, zorder=4)

    # log scale and custom ticks
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())

    ax.set_xlabel("Network size N")
    ax.set_ylabel("Max memory lifetime (mean ± std over runs)")
    ax.legend()
    fig.tight_layout()
    plt.show()


Ns = [200, 400, 800, 1600, 3500]
synapse_specs = {
    "None":   {},
    "BfLin":  {"bf_m":4, "bf_levels":35, "bf_alpha":0.25,
               "bf_decreasing_levels":True, "bf_beta":2},
    "Wdecay": {"hebb_lambda":0.995, "hebb_alpha":4},
    "DoubleW": {
        200:  {"DW_r1":0.02, "DW_r2":1.0, "DW_r3":0, "DW_C":1.3},
        400:  {"DW_r1":0.01, "DW_r2":1.0, "DW_r3":0, "DW_C":1.8},
        800:  {"DW_r1":0.01, "DW_r2":1.0, "DW_r3":0, "DW_C":1.2},
        1600: {"DW_r1":0.01, "DW_r2":1.0, "DW_r3":0, "DW_C":0.7},
        3500: {"DW_r1":0.01, "DW_r2":1.0, "DW_r3":0, "DW_C":0.35},
    }
}
max_t_dict = {
    "None":    [70, 125, 200 , 350, 650],
    "BfLin":   [50,  60,  70, 100, 175],
    "Wdecay":  [50,  70, 120, 160, 200],
    "DoubleW": [60, 150, 250, 700, 1300],
}
threshold = 0.97
runs = 5
iters_per_store = 10
coding = "dense"
f = 0.5

# Generate the plot
plot_max_memory_lifetime(
    Ns,
    synapse_specs,
    max_t_dict,
    threshold=threshold,
    runs=runs,
    iters_per_store=iters_per_store,
    coding=coding,
    f=f
)

# %%
