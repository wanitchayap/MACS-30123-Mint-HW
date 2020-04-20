from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time

import numpy as np
import scipy.stats as sts

def sim_random_shock():
    
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Set model parameters
    rho = 0.5
    mu = 3.0
    sigma = 1.0
    z_0 = mu

    # Set simulation parameters, draw all idiosyncratic random shocks,
    # and create empty containers
    S = 1000 # Set the number of lives to simulate
    T = int(4160) # Set the number of periods for each simulation
    np.random.seed(25)
    
    # number of simulations per each core
    N = int(S/size)
    
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
    z_mat = np.zeros((T, N))
    z_mat[0, :] = z_0

    for s_ind in range(N):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    z_all = None
    if rank == 0:
        z_all = np.empty([T, S], dtype='float')
    comm.Gather(sendbuf = z_all, recvbuf = z_all, root=0)

    if rank == 0:
        time_elapsed = time.time() - t0
        print(time_elapsed)
        return time_elapsed

    return

def main():
    sim_random_shock()

if __name__ == '__main__':
    main()
