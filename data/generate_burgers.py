import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
'''
This file contains six functions:

- generate_initial_y(x): Generate the initial state y(0, x).
- generate_control_sequence(x, t): Generate control sequence u(t, x).
- burgers_update(y, u, dx, dt, nu=0.01): Return the updated state vector y using the Lax-Friedrichs algorithm with viscosity.
- get_trajectory(y0, nt = 500, dt= 0.001, nx = 128, dx = 10/128): Generate a single trajectory of the Burgers equation.
- load_burgers(): Generate multiple trajectories of the Burgers equation.
- visualize_state_trajectory(): Visualize the state trajectory with animation.

The most important ones are burgers_update(), get_trajectory() and load_burgers().

You can also direcly run this file, which will show an animation of the state evolution. You can also import the functions from elsewhere.
'''


def burgers_update(y, u, dx, dt, nu=0.01):
    r"""
    Given current state y and control signal u at some time t, return the updated state vector y at time t+dt.
    The boundary condition is 0.

    :param y: (n,), state vector.
    :param u: (m,), control signal.
    :param dx: the spatial step size.
    :param dt: the temporal step size.
    :param nu: kinematic viscosity.

    :return: (n,), the updated state vector.
    """
    assert type(y) == np.ndarray and type(u) == np.ndarray, "y and u must be numpy arrays"
    assert y.ndim == 1 and u.ndim == 1, "y and u must be 1 dimensional arrays"
    assert y.shape == u.shape, "y and u must have the same shape"
    
    y_new = np.copy(y)
    ya=np.roll(y,-1) # Equivalent to y_{i+1}
    yb=np.roll(y,1) # Equivalent to y_{i-1}
    convective_term = (ya**2/2 - yb**2/2) / (2*dx) # Convective term (advection)
    diffusive_term = nu*(ya - 2*y + yb) / (dx**2) # Diffusive term (viscosity)
    y_new = y_new - dt*convective_term + dt*diffusive_term + u*dt # Lax-Friedrichs update with viscosity
    y_new[0] = 0 # Apply 0 boundary conditions
    y_new[-1] = 0 # Apply 0 boundary conditions
    return y_new


def generate_initial_y(x):
    r"""
    Generate the initial state u(0, x) as a superposition of two Gaussian functions.
    """
    # Sampling parameters for the Gaussian functions
    a1 = np.random.uniform(0, 2)
    a2 = np.random.uniform(-2, 0)
    mu1 = np.random.uniform(0.2, 0.4)
    mu2 = np.random.uniform(0.6, 0.8)
    sigma1 = np.random.uniform(0.05, 0.15)
    sigma2 = np.random.uniform(0.05, 0.15)
    
    y = a1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2)) +\
        a2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    #y[0] = 0.0 # Apply 0 boundary condition
    #y[-1] = 0.0 # Apply 0 boundary condition
    return y


def generate_control_sequence(x, t):
    r"""
    Generate control sequence u(t, x) as a superposition of 8 Gaussian functions.
    """
    u = np.zeros_like(x)
    for i in range(8):
        if i==0:
            ai = np.random.uniform(-1.5,1.5)
        else:
            if random.choice([True, False])==True: # 50% chance to add a new Gaussian function
                ai = np.random.uniform(-1.5,1.5)
            else:
                ai = 0
        b1_i = np.random.uniform(0, 1)
        b2_i = np.random.uniform(0, 1)
        sigma1_i = np.random.uniform(0.05, 0.2)
        sigma2_i = np.random.uniform(0.05, 0.2)
        u += ai * np.exp(-((x - b1_i)**2) / (2 * sigma1_i**2)) * np.exp(-((t - b2_i)**2) / (2 * sigma2_i**2))
    #u*=0.1 # Scale the control signal to a proper range (maybe not necessary)
    return u


def get_trajectory(
        y0,
        nt = 100, # Number of time steps
        dt= 0.01, # Temporal interval
        nx = 128, # Number of spatial nodes (grid points)
        dx = 1/128, # Spatial interval
        ):
    r'''
    Generate a single trajectory of the Burgers equation.

    nx: number of spatial nodes (state dim)
    nt: number of time steps
    dx: spatial interval
    dt: temporal interval
    y0: (state_dim,), initial state vector
    '''
    def roughness(y):
        r'''
        The average absolute difference between adjacent points in the input vector.
        '''
        assert type(y) == np.ndarray, "y must be a numpy array"
        assert y.ndim == 1, "y must be a 1 dimensional array"
        return np.sum(np.abs(y[1:] - y[:-1])) / (y.shape[0] - 1)

    y_list = [] # Holds states from y_0 to y_{nt}
    u_list = [] # Holds control sequence u_0 to u_{nt-1}
    
    y = y0 # Set the initial state
    y_list.append(y) # Append the initial state to Y_bar

    for t_idx in range(nt): # Iterate over time steps
        u=generate_control_sequence(y,t_idx*dt) # (n,), Control vector at time t_idx*dt
        y_new=burgers_update(y, u, dx, dt) # (n,), Updated state vector at time t_idx*dt, with control signal u
        u_list.append(u) # Append u_{t_idx}
        y_list.append(y_new) # Append y_bar_{t_idx+1}
        y=y_new # Update y
    
    # Set the observations, actions, and final state
    state_trajectory=np.stack(y_list[:-1]) # (nt, state_dim), States from y_0 to y_{nt-1}
    action_trajectory=(np.stack(u_list)) # (nt, state_dim), Control sequence from u_0 to u_{nt-1}
    final_state=y # (state_dim,), Final state y_{nt}
    
    # Set timeout flags
    # Note: this flag may not be necessary in practice. No need to pay too much attention to this part.
    timeouts = np.zeros(nt, dtype=bool) # (nt,), First set all time steps as False (non-timeout)
    timeouts[-1] = True # Set the last time step as timeout
    
    # Set terminal flags
    # Note: this flag may not be necessary in practice. No need to pay too much attention to this part.
    terminals = np.zeros(nt, dtype=bool) # (nt,), First set all time steps as False (non-terminal)
    for t_idx in range(nt): # Iterate over time steps
        if roughness(state_trajectory[t_idx])/dx > 10: # If the roughness is so large that the average first derivative among all spatial nodes is greater that 10, set it as terminal
            terminals[t_idx:] = True # Set all subsequent time steps as terminal
            break # Break the loop over time steps, since all subsequent time steps have been set as terminal
    
    # Set the reward values
    # Note: The rewards may be calculated elsewhere by other means. No need to pay too much attention to this part.
    rewards = np.zeros(nt) # (nt,), Initialize the reward values at all time steps as 0
    if True not in terminals: # If the trajectory converges (doesn't terminate early), 
        rewards = -((state_trajectory - final_state)**2).mean(axis=-1) # Set the reward as the negative L2 distance between the final state and the current state
    else: # If the trajectory diverges (terminates early)
        terminal_timestep=list(terminals).index(True) # Get the index of the first terminal time step
        roughness_turn_timestep=None # The last time step at which the roughness turns from going smaller to going larger (will be set later)
        for t_idx in reversed(range(1, terminal_timestep)): # Iterate over time steps
            if t_idx < nt/2:
                break
            if roughness(state_trajectory[t_idx+1])/dx < \
                    roughness(state_trajectory[t_idx])/dx: # Find the last time step at which the roughness turns from going smaller to going larger
                final_state=state_trajectory[t_idx] # (n,); The final state is reset to this time step
                roughness_turn_timestep=t_idx # The roughness turn time step is set to this time step
                break
        rewards[:roughness_turn_timestep] = -((state_trajectory[:roughness_turn_timestep] - final_state)**2).mean(axis=-1)
            # The rewards before the terminal time step are set as the negative L2 distance between the final state and the current state
        rewards[roughness_turn_timestep:] = -np.inf # The rewards after the terminal time step are set as -inf
    
    # Add the action reward to the total reward
    # Note: The action reward may be calculated elsewhere by other means. No need to pay too much attention to this part.
    action_rewards = -(action_trajectory**2).mean(axis=-1) # Set the action reward as the L2 norm of the control sequence at each time step
    rewards += action_rewards
    return state_trajectory, action_trajectory, final_state, timeouts, terminals, rewards # Shape: (nt, state_dim), (nt, state_dim), (state_dim,), (nt,), (nt,), (nt,)


def load_burgers(
        x_range=(0,1), # Spatial grid domain of the burgers equation
        nt = 500, # Number of time steps
        nx = 128, # Number of spatial nodes (grid points)
        dt= 0.001, # Temporal interval
        N = 3, # Number of samples (trajectories) to generate
        ):
    r'''
    Load several trajectories of the Burgers equation.

    x_range: the spatial grid domain of the burgers equation
    nt: number of time steps
    nx: number of spatial nodes (grid points)
    dt: temporal interval
    N: number of samples (trajectories) to generate
    '''
    dx = (x_range[1]-x_range[0])/nx # Calculate the spatial interval (assume equal spacing)
    x = np.linspace(*x_range, nx) # Initialize the spatial grid
    n = nx # state dimension. In this case, the state is all function values at the spatial grid points, so n=nx.
    
    Y_bar_list = [] # Holds states trajectories of N samples
    Y_f_list = [] # Holds final state of N samples
    U_list = [] # Holds control sequence trajectories of N samples
    timeout_list = [] # Holds timeout flags of N samples
    terminal_list = [] # Holds terminal flags of N samples
    reward_list = [] # Holds reward values of N samples

    # Simulate the system for N samples
    for _ in tqdm.tqdm(range(N), desc="Generating samples"):
        y0=generate_initial_y(x) # Set the initial condition
        state_trajectory, action_trajectory, final_state, timeouts, terminals, rewards=get_trajectory(y0, nt=nt, dt=dt, nx=nx, dx=dx) # Generate a single trajectory

        Y_bar_list.append(state_trajectory) # Append the state trajectory to Y_bar_list
        Y_f_list.append(final_state) # Append the final state to Y_f_list
        U_list.append(action_trajectory) # Append the control sequence to U_list
        timeout_list.append(timeouts) # Append the timeout flags to timeout_list
        terminal_list.append(terminals) # Append the terminal flags to terminal_list
        reward_list.append(rewards) # Append the reward values to reward_list
        
    # Save the data into a dictionary
    data_dict = { # d4rl API
            'observations': np.stack(Y_bar_list), # (N, nt, n)
            'actions': np.stack(U_list), # (N, nt, n)
            'rewards': np.stack(reward_list),# (N, nt)
            'terminals': np.stack(terminal_list), # (N, nt)
            'timeouts': np.stack(timeout_list), # (N, nt)
            'Y_f': np.stack(Y_f_list), # (N, n)
            'meta_data': {
                'num_nodes': nx, 
                'input_dim': nx, 
                'control_horizon': nt-1,
                'num_samples': N,
            },
        }
    print("Observations shape: ", data_dict['observations'].shape) # (N, nt, n)
    print("Y_f shape: ", data_dict['Y_f'].shape) # (N, n)
    print("Actions shape: ", data_dict['actions'].shape) # (N, nt, n)
    print("Terminals shape: ", data_dict['terminals'].shape) # (N, nt)
    print("Timeouts shape: ", data_dict['timeouts'].shape) # (N, nt)
    print("Rewards shape: ", data_dict['rewards'].shape) # (N, nt)
    
    return data_dict

def visualize_state_trajectory(x, Y, dt, nt, frame_interval=20):
    r"""
    Visualize the state trajectories of a sample.
    The total number of frames is nt.

    :param x: (nx,), the spatial grid.
    :param Y: (T, nx), the state trajectory.
    :param dt: the temporal interval.
    :param nt: the number of time steps.
    :param frame_interval: the frame interval (in milliseconds) of the animation.
    """
    from matplotlib.animation import FuncAnimation

    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(9,6))
    line, = ax.plot(x, Y[0], label="Burgers equation solution")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(x.min(), x.max())
    ax.set_title("Burgers Equation Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    time_text = ax.text(0.5, 0.95, "",
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

    # Animation update function
    def update(frame):
        line.set_ydata(Y[frame])
        time_text.set_text(f'Current Time: {frame * dt:.3f}/{nt*dt:.3f}')  # 更新时间提示
        return line, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=frame_interval)
    plt.legend()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------


# The following code is a specific implementation of code used in the main script. The logic is basically the same as get_trajectory().
def get_sampled_trajectory(
        y0,
        nt = 10000, # Number of time steps
        dt= 0.01, # Temporal interval
        nx = 128, # Number of spatial nodes (grid points)
        dx = 1/128, # Spatial interval
        t_sample_interval=1000, # The interval of time steps to generate a new control signal
        ):
    r'''
    Generate a sampled trajectory of the Burgers equation.
    Specifically, generate 10000 timesteps, with an interval of 0.0001. Change the control signal every 1000 timesteps. 

    :param y0: (state_dim,), initial state vector
    :param nx: number of spatial nodes (state dim)
    :param nt: number of time steps
    :param dx: spatial interval
    :param dt: temporal interval
    :param t_sample_interval: the interval of time steps to generate a new control signal

    :return: state_trajectory, action_trajectory, final_state
    '''

    y_list = [] # Holds states from y_0 to y_{nt}
    u_list = [] # Holds control sequence u_0 to u_{nt-1}
    
    y = y0 # Set the initial state
    y_list.append(y) # Append the initial state to Y_bar

    for t_idx in range(nt): # Iterate over time steps
        if t_idx % t_sample_interval == 0: # Generate a new control signal every 1000 time steps
            u=generate_control_sequence(y,t_idx*dt) # (n,), Control vector at time t_idx*dt
        # Within every 1000 time steps, the control signal does not change.
        y_new=burgers_update(y, u, dx, dt) # (n,), Updated state vector at time t_idx*dt, with control signal u
        u_list.append(u) # Append u_{t_idx}
        y_list.append(y_new) # Append y_bar_{t_idx+1}
        y=y_new # Update y
    
    # Set the observations, actions, and final state
    state_trajectory=np.stack(y_list[0:-1:t_sample_interval]) # (nt, state_dim), States from y_0 to y_{nt-1}, every 1000 time steps
    action_trajectory=(np.stack(u_list[::t_sample_interval])) # (nt, state_dim), Control sequence from u_0 to u_{nt-1}, every 1000 time steps
    final_state=y # (state_dim,), Final state y_{nt}
    
    return state_trajectory, action_trajectory, final_state


# The following code is a specific implementation of code used in the main script. The logic is basically the same as load_burgers().
def load_burgers_data_sampled(
        x_range=(0,1), # Spatial grid domain of the burgers equation
        nt = 10000, # Number of time steps
        nx = 128, # Number of spatial nodes (grid points)
        dt= 0.0001, # Temporal interval
        t_sample_interval=1000, # The interval of time steps to generate a new control signal
        N = 3, # Number of samples (trajectories) to generate
        save_dir=None, # Directory to save the data
        ):
    r'''
    Load several trajectories of the Burgers equation.

    x_range: the spatial grid domain of the burgers equation
    nt: number of time steps
    nx: number of spatial nodes (grid points)
    dt: temporal interval
    t_sample_interval: the interval of time steps to generate a new control signal
    N: number of samples (trajectories) to generate
    '''
    dx = (x_range[1]-x_range[0])/nx # Calculate the spatial interval (assume equal spacing)
    x = np.linspace(*x_range, nx) # Initialize the spatial grid
    n = nx # state dimension. In this case, the state is all function values at the spatial grid points, so n=nx.
    nt1 = nt//t_sample_interval # Number of time steps per trajectory (sampled)
    print(f"Number of time steps per trajectory: {nt1}")
    Y_bar_list = [] # Holds states trajectories of N samples
    Y_f_list = [] # Holds final state of N samples
    U_list = [] # Holds control sequence trajectories of N samples
    Y_bar_next_list = [] # Holds next state of N samples

    # Simulate the system for N samples
    for _ in tqdm.tqdm(range(N), desc="Generating trajectories"):
        y0=generate_initial_y(x) # Set the initial condition
        state_trajectory, action_trajectory, final_state=get_sampled_trajectory(y0, nt=nt, dt=dt, nx=nx, dx=dx, t_sample_interval=t_sample_interval) # Generate a single trajectory

        Y_bar_list.append(state_trajectory) # Append the state trajectory to Y_bar_list
        Y_f_list.append(final_state) # Append the final state to Y_f_list
        U_list.append(action_trajectory) # Append the control sequence to U_list
        Y_bar_next_list.append(np.concatenate([state_trajectory[1:,:], final_state[np.newaxis,:]], axis=0)) # Append the next state to Y_bar_next_list
    
    # data_name = f'burgers_{N}_{nt1}_{n}_{timestr}_new2.pkl'
    data_dict = {}
    data_dict['sys'] = {"C": np.identity(n)}  # Assuming the system matrix C is identity
    data_dict['meta_data'] = {
        'num_nodes': n, 
        'input_dim': n, 
        'control_horizon': nt1, 
        'num_samples': N,
        'time interval': dt,
        'num time steps': nt,
        'x_range': x_range,
    }
    # data_dict['data'] = {'Y_bar': np.stack(Y_bar_list), 'Y_f': np.stack(Y_f_list), 'U': np.flip(np.stack(U_list), axis=1)}
    data_dict['data'] = {'Y_bar': np.stack(Y_bar_list), 'Y_f': np.stack(Y_f_list), 'U': np.stack(U_list)}
    # split the data into train, valid, test
    U_3d = data_dict['data']['U']
    Y_bar_3d = data_dict['data']['Y_bar']
    Y_f_3d = data_dict['data']['Y_f']
    indices = np.random.choice(U_3d.shape[0], size=int(U_3d.shape[0]*0.002), replace=False)
    data_dict['train_data'] = {}
    data_dict['val_data'] = {}
    data_dict['test_data'] = {}
    for key,data in [('U', U_3d), ('Y_bar', Y_bar_3d), ('Y_f', Y_f_3d)]:
        valid_data = data[indices[:int(indices.shape[0]/2)]]
        test_data = data[indices[int(indices.shape[0]/2):]]
        train_data = np.delete(data, indices, axis=0)
        print(train_data.shape, valid_data.shape, test_data.shape)
        data_dict['train_data'][key] = train_data
        data_dict['val_data'][key] = valid_data
        data_dict['test_data'][key] = test_data

    # print statistics
    for key in ['U', 'Y_bar', 'Y_f']:
        print(key)
        print('train:', data_dict['train_data'][key].shape)
        print('valid:', data_dict['val_data'][key].shape)
        print('test:', data_dict['test_data'][key].shape)

    del data_dict['data']
    
    # # Save the data into a dictionary
    # data_dict = { # d4rl API
    #         'observations': np.stack(Y_bar_list), # (N, nt1, n)
    #         'actions': np.stack(U_list), # (N, nt1, n)
    #         'Y_f': np.stack(Y_f_list), # (N, n)
    #         'next_observations': np.stack(Y_bar_next_list), # (N, nt1, n)
    #         'meta_data': {
    #             'spatial domain': x_range,
    #             'num_nodes': nx,
    #             'input_dim': nx,
    #             'time interval': dt,
    #             'num time steps': nt,
    #             'num trajectories': N,
    #             'time step sample interval': t_sample_interval,
    #             'num samples per trajectory': nt1,
    #         },
    #     }
    # print("Next observations shape: ", data_dict['next_observations'].shape) # (N, nt1, n)

    if save_dir is not None:
        import os
        import pickle
        import time
        if not os.path.exists(save_dir):
            print(f"Directory {save_dir} does not exist. Creating it...")
            os.makedirs(save_dir)
            print(f"Created directory {save_dir}")
        timestr = str(int(time.time()))
        data_name = f'burgers_{N}_{nt1}_{n}_{timestr}_split.pkl'
        save_path = os.path.join(save_dir, data_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
            print(f"Burgers data saved to {save_path}")

    return data_dict

# ---------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    load_burgers_data_sampled(N=20000,save_dir='./synthetic_data')


# -------------------test----------------------------------------------
# You can directly run this file for testing. Jupyter notebook may not work, .py is recommended.


#if __name__ == '__main__':
#    # Hyperparameters (can be adjusted manually)
#    x_range=(0,1) # Spatial grid domain of the burgers equation
#    nt=10000 # Number of time steps
#    dt=1e-4 # Temporal interval
#    nx=128 # Number of spatial nodes (grid points)
#    dx=1/128 # Spatial interval
#    frame_interval=2 # Frame interval of the animation (in milliseconds)
#
#    # Generate a single trajectory and visualize it
#    x=np.linspace(*x_range,nx) # Spatial grid for the Burgers equation
#    y0=generate_initial_y(x)
#    state_trajectory, _, _, _, _, _ = get_trajectory(y0,nt=nt,dt=dt,nx=nx,dx=(x_range[1]-x_range[0])/nx)
#    visualize_state_trajectory(x, state_trajectory, dt, nt, frame_interval) # Show the animation of the state evolution.
#    
#    # Below are functions for testing and debugging.
#    def visualize_node_values(state_trajectory):
#        '''
#        A test function.
#        Visualize the value evolution at some certain spatial nodes.
#        '''
#        plt.figure(figsize=(9,6))
#        plt.title("Burgers equation solution at certain spatial nodes")
#        plt.xlabel("t")
#        plt.ylabel("u(x,t)")
#        plt.plot(state_trajectory[:,16], label="Node 16")
#        plt.plot(state_trajectory[:,32], label="Node 32")
#        plt.plot(state_trajectory[:,64], label="Node 64")
#        plt.plot(state_trajectory[:,96], label="Node 96")
#        plt.plot(state_trajectory[:,112], label="Node 112")
#        plt.legend()
#        plt.show()
#    
#    def visualize_curves_at_different_time_steps(state_trajectory):
#        '''
#        A test function.
#        Visualize the curves at different time steps (in a static plot)
#        '''
#        plt.figure(figsize=(9,6))
#        plt.title("Burgers equation solution at different time steps")
#        plt.xlabel("x")
#        plt.ylabel("u(x,t)")
#        plt.plot(state_trajectory[0,:], label="Initial state")
#        plt.plot(state_trajectory[1000,:], label="1000th time step")
#        plt.plot(state_trajectory[2000,:], label="2000th time step")
#        plt.plot(state_trajectory[3000,:], label="3000th time step")
#        plt.plot(state_trajectory[4000,:], label="4000th time step")
#        plt.plot(state_trajectory[5000,:], label="5000th time step")
#        plt.plot(state_trajectory[6000,:], label="6000th time step")
#        plt.plot(state_trajectory[7000,:], label="7000th time step")
#        plt.plot(state_trajectory[8000,:], label="8000th time step")
#        plt.plot(state_trajectory[9000,:], label="9000th time step")
#        plt.plot(state_trajectory[-1,:], label="Final state")
#        plt.legend()
#        plt.show()
#    
#    visualize_node_values(state_trajectory)
#    visualize_curves_at_different_time_steps(state_trajectory)