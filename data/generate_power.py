import numpy as np
import pickle
import tqdm
# NE_TEST_PARAMETERS - Parameters of New England 39-bus 10-generator Test case
# from Y. Susuki, I. Mezic, T. Hikihara, "Coherent Swing Instability of Power Grids", J Nonlinear Sci, 2011

# Initial phase angles at t=0 (generators' order: 10, 2, 3, 4, 5, 6, 7, 8, 9)


y0 = np.array([
    0.0193, 0.1757, 0.1998, 0.1824, 0.3329,
    0.2016, 0.2042, 0.1844, 0.3145, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

# Inertia matrix / pi / 60 Hz
H = np.diag([
    0.2228, 0.1607, 0.1899, 0.1517, 0.1379,
    0.1846, 0.1401, 0.1289, 0.1830, 2.6526
])

# Mechanical input power
P = np.array([
    1.2500, 2.4943, 3.2500, 3.1600, 2.5400,
    3.2500, 2.8000, 2.7000, 4.1500, 5.0000
])

# Damping matrix
D = np.diag([0.0050] * 10)

# Terminal voltages
E_complex = np.array([
    1.0568 + 0.0204j, 0.9974 + 0.1770j, 0.9842 + 0.1993j,
    0.9804 + 0.1809j, 1.0855 + 0.3753j, 1.0569 + 0.2160j,
    1.0423 + 0.2158j, 0.9742 + 0.1817j, 0.9352 + 0.3043j,
    1.0205 - 0.0431j
])
E = np.abs(E_complex)

# Network impedance matrix G + i*B (pre-fault)
Y3 = np.array([
    [0.7239 - 15.0009j, 0.2080 + 0.9484j, 0.2536 + 1.2183j, 0.2565 + 1.2209j, 0.1033 + 0.4161j, 0.2348 + 1.0950j, 0.1950 + 0.9237j, 0.0670 + 2.9064j, 0.1961 + 1.5928j, 0.6099 + 4.8881j],
    [0.2080 + 0.9484j, 0.2519 - 9.1440j, 0.2603 + 1.8170j, 0.1657 + 0.6891j, 0.0655 + 0.2346j, 0.1513 + 0.6180j, 0.1259 + 0.5214j, 0.0916 + 0.5287j, 0.1150 + 0.4400j, 0.4159 + 2.7502j],
    [0.2536 + 1.2183j, 0.2603 + 1.8170j, 0.3870 - 10.9096j, 0.2142 + 0.9763j, 0.0857 + 0.3326j, 0.1959 + 0.8756j, 0.1629 + 0.7386j, 0.1144 + 0.6848j, 0.1471 + 0.5888j, 0.4569 + 2.9961j],
    [0.2565 + 1.2209j, 0.1657 + 0.6891j, 0.2142 + 0.9763j, 0.8131 - 12.0737j, 0.2843 + 1.9774j, 0.3178 + 1.7507j, 0.2633 + 1.4766j, 0.1608 + 0.7478j, 0.2104 + 0.8320j, 0.3469 + 1.6513j],
    [0.1033 + 0.4161j, 0.0655 + 0.2346j, 0.0857 + 0.3326j, 0.2843 + 1.9774j, 0.1964 - 5.5114j, 0.1309 + 0.5973j, 0.1088 + 0.5038j, 0.0645 + 0.2548j, 0.0826 + 0.2831j, 0.1397 + 0.5628j],
    [0.2348 + 1.0950j, 0.1513 + 0.6180j, 0.1959 + 0.8756j, 0.3178 + 1.7507j, 0.1309 + 0.5973j, 0.4550 - 11.1674j, 0.3366 + 3.1985j, 0.1471 + 0.6707j, 0.1920 + 0.7461j, 0.3175 + 1.4810j],
    [0.1950 + 0.9237j, 0.1259 + 0.5214j, 0.1629 + 0.7386j, 0.2633 + 1.4766j, 0.1088 + 0.5038j, 0.3366 + 3.1985j, 0.4039 - 9.6140j, 0.1223 + 0.5657j, 0.1599 + 0.6294j, 0.2638 + 1.2493j],
    [0.0670 + 2.9064j, 0.0916 + 0.5287j, 0.1144 + 0.6848j, 0.1608 + 0.7478j, 0.0645 + 0.2548j, 0.1471 + 0.6707j, 0.1223 + 0.5657j, 0.6650 - 10.0393j, 0.3225 + 1.2618j, 0.0991 + 2.5318j],
    [0.1961 + 1.5928j, 0.1150 + 0.4400j, 0.1471 + 0.5888j, 0.2104 + 0.8320j, 0.0826 + 0.2831j, 0.1920 + 0.7461j, 0.1599 + 0.6294j, 0.3225 + 1.2618j, 0.9403 - 7.5882j, 0.2377 + 1.5792j],
    [0.6099 + 4.8881j, 0.4159 + 2.7502j, 0.4569 + 2.9961j, 0.3469 + 1.6513j, 0.1397 + 0.5628j, 0.3175 + 1.4810j, 0.2638 + 1.2493j, 0.0991 + 2.5318j, 0.2377 + 1.5792j, 5.9222 - 18.6157j]
])

# Network impedance matrix (fault on)
Y32 = np.array([
    [0.5383 - 15.7638j, 0.0901 + 0.5182j, 0.0994 + 0.6084j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.0490 + 2.4392j, 0.0472 + 1.0736j, 0.3589 + 3.8563j],
    [0.0901 + 0.5182j, 0.1779 - 9.3864j, 0.1628 + 1.4731j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0180 + 0.2653j, 0.0219 + 0.1476j, 0.2564 + 2.1683j],
    [0.0994 + 0.6084j, 0.1628 + 1.4731j, 0.2591 - 11.3971j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0181 + 0.3113j, 0.0241 + 0.1739j, 0.2483 + 2.1712j],
    [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.4671 - 14.0254j, 0.1411 + 1.3115j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
    [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.1411 + 1.3115j, 0.1389 - 5.7383j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
    [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.1633 - 12.7378j, 0.0947 + 1.8739j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
    [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0947 + 1.8739j, 0.2035 - 10.7312j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
    [-0.0490 + 2.4392j, 0.0180 + 0.2653j, 0.0181 + 0.3113j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.5925 - 10.3254j, 0.2297 + 0.9440j, -0.0579 + 1.8999j],
    [0.0472 + 1.0736j, 0.0219 + 0.1476j, 0.0241 + 0.1739j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.2297 + 0.9440j, 0.8235 - 7.9409j, 0.0363 + 0.8770j],
    [0.3589 + 3.8563j, 0.2564 + 2.1683j, 0.2483 + 2.1712j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.0579 + 1.8999j, 0.0363 + 0.8770j, 5.5826 - 20.0113j]
])

# Network impedance matrix (post-fault)
Y33 = np.array([
    [0.8012 - 14.3511j, 0.2163 + 0.9784j, 0.2559 + 1.1997j, 0.1629 + 0.5591j, 0.0629 + 0.1900j, 0.1483 + 0.5013j, 0.1237 + 0.4230j, 0.1385 + 3.3322j, 0.3015 + 2.1485j, 0.6576 + 5.3495j],
    [0.2163 + 0.9784j, 0.2525 - 9.1427j, 0.2603 + 1.8161j, 0.1565 + 0.6587j, 0.0619 + 0.2243j, 0.1429 + 0.5908j, 0.1189 + 0.4984j, 0.0980 + 0.5482j, 0.1239 + 0.4653j, 0.4214 + 2.7715j],
    [0.2559 + 1.1997j, 0.2603 + 1.8161j, 0.3868 - 10.9091j, 0.2124 + 0.9954j, 0.0853 + 0.3392j, 0.1943 + 0.8927j, 0.1615 + 0.7530j, 0.1153 + 0.6724j, 0.1479 + 0.5726j, 0.4586 + 2.9829j],
    [0.1629 + 0.5591j, 0.1565 + 0.6587j, 0.2124 + 0.9954j, 0.9236 - 11.4000j, 0.3306 + 2.2074j, 0.4194 + 2.3551j, 0.3474 + 1.9863j, 0.0782 + 0.3146j, 0.0903 + 0.2669j, 0.2878 + 1.1812j],
    [0.0629 + 0.1900j, 0.0619 + 0.2243j, 0.0853 + 0.3392j, 0.3306 + 2.2074j, 0.2151 - 5.4330j, 0.1734 + 0.8035j, 0.1440 + 0.6778j, 0.0308 + 0.1071j, 0.0343 + 0.0905j, 0.1135 + 0.4020j],
    [0.1483 + 0.5013j, 0.1429 + 0.5908j, 0.1943 + 0.8927j, 0.4194 + 2.3551j, 0.1734 + 0.8035j, 0.5485 - 10.6253j, 0.4139 + 3.6557j, 0.0714 + 0.2821j, 0.0820 + 0.2392j, 0.2627 + 1.0592j],
    [0.1237 + 0.4230j, 0.1189 + 0.4984j, 0.1615 + 0.7530j, 0.3474 + 1.9863j, 0.1440 + 0.6778j, 0.4139 + 3.6557j, 0.4679 - 9.2284j, 0.0594 + 0.2380j, 0.0685 + 0.2019j, 0.2187 + 0.8936j],
    [0.1385 + 3.3322j, 0.0980 + 0.5482j, 0.1153 + 0.6724j, 0.0782 + 0.3146j, 0.0308 + 0.1071j, 0.0714 + 0.2821j, 0.0594 + 0.2380j, 0.7257 - 9.7609j, 0.4096 + 1.6248j, 0.1451 + 2.8344j],
    [0.3015 + 2.1485j, 0.1239 + 0.4653j, 0.1479 + 0.5726j, 0.0903 + 0.2669j, 0.0343 + 0.0905j, 0.0820 + 0.2392j, 0.0685 + 0.2019j, 0.4096 + 1.6248j, 1.0644 - 7.1152j, 0.3063 + 1.9743j],
    [0.6576 + 5.3495j, 0.4214 + 2.7715j, 0.4586 + 2.9829j, 0.2878 + 1.1812j, 0.1135 + 0.4020j, 0.2627 + 1.0592j, 0.2187 + 0.8936j, 0.1451 + 2.8344j, 0.3063 + 1.9743j, 5.9509 - 18.2881j]
])

# Generate data for control
N = 100000  # number of samples
sigma = 0.01  # variance of initial states
U = []
X0 = []
X1T = []
XT = []
X0T = []
# Time parameters
delta_t = 0.01  # time step (you may adjust as needed)
T = 0.31  # total simulation time (you may adjust as needed)
T_fault_in = 2  # fault start time
T_fault_end = 2.31  # fault end time

m = 9  # number of inputs
n = 18  # number of states

def swing(t, delta_t, y, u):
    # Simulate power generators swing dynamics
    global H, P, D, E, Y3, Y32, T_fault_in, T_fault_end

    yp = np.zeros(18)

    # Select the appropriate network impedance matrix
    if t < T_fault_in:
        Y = Y3
    elif T_fault_in <= t < T_fault_end:
        Y = Y32
    else:
        Y = Y3

    # Update phase angles
    for i in range(9):
        yp[i] = delta_t * y[9 + i] + y[i] + u[i]

    k=[]
    for ki in range(9):
        idxk = ki + 1  # Adjusting for zero-based indexing
        sum_termsk = 0
        for jk in range(10):
            if jk != idxk:
                sum_termsk += E[idxk] * E[jk] * (
                    np.real(Y[idxk, jk])
                )
        k.append(delta_t * (1 / H[idxk, idxk]) * (
            P[idxk] - np.real(Y[idxk, idxk]) * E[idxk] ** 2 - sum_termsk
        ))
    # Update angular velocities
    for i in range(9):
        idx = i + 1  # Adjusting for zero-based indexing
        sum_terms = 0
        for j in range(10):
            if j != idx:
                y2 = y[j-1] if j!=i else 0
                angle_diff = y[i] - y2
                sum_terms += E[idx] * E[j] * (
                    np.real(Y[idx, j]) * np.cos(angle_diff) +
                    np.imag(Y[idx, j]) * np.sin(angle_diff)
                )

        yp[9 + i] = (
            delta_t * (1 / H[idx, idx]) * (
                -D[idx, idx] * (y[9 + i] + u[i]) +
                P[idx] -
                np.real(Y[idx, idx]) * E[idx] ** 2 -
                sum_terms
            ) + y[9 + i]-k[i]
        )

    return yp

if __name__ =='__main__':
    sigma2 = 0.1
    progress = tqdm.tqdm(range(N))
    for i in progress:
        k = 0

        # Random input
        u = sigma2 * np.random.randn(m, int(round(T / delta_t)))

        # Random initial state
        y0 = sigma * np.random.randn(n)
        y0_data = y0.copy()

        # Preallocate y_data
        time_steps = int(round(T / delta_t))
        y_data = np.zeros((n, time_steps))

        for t_idx, t in enumerate(np.arange(0, T, delta_t)):
            y_data[:, k] = swing(t, delta_t, y0_data, u[:, k])
            y0_data = y_data[:, k]
            k += 1

        U.append(u.T) # dimensions: (time_steps, m)
        X0.append(y0)
        X1T.append(y_data[:, :time_steps - 1].T)
        X0T.append(np.concatenate((np.expand_dims(y0, axis=0), y_data[:, :time_steps - 1].T), axis=0))
        XT.append(y_data[:, -1])

    U = np.stack(np.flipud(U)) # dimensions: (N, time_steps, m), the fliped axis is the time axis
    X0 = np.stack(X0)
    X1T = np.stack(X1T)
    XT = np.stack(XT)
    X0T = np.stack(X0T)

    print('U shape:', U.shape, 'X0 shape:', X0.shape, 'X0T shape:', X0T.shape, 'XT shape:', XT.shape)

    T = X0T.shape[1] + 1

    Y_bar = X0T
    Y_f = XT
    # save data
    data_name = f'power_{n}_{m}_{T-1}_{N}_sigma={sigma}_{sigma2}.pkl'
    data_dict = {}
    data_dict['sys'] = { "C": np.identity(n)}
    # data_dict['adj'] = np.copy(A)
    data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'control_horizon': T-1, 'num_samples': N}
    print(data_dict['meta_data'])
    data_dict['data'] = {'Y_bar': Y_bar, 'Y_f': Y_f, 'U': U}
    with open('/data2/chenhongyi/diffcon/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
        pickle.dump(data_dict, f)