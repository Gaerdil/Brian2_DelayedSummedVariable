from brian2 import *
import time
import datetime

"""
This implements a model with delayed summed variable: 
- the variable v evolves in input NeuronGroup G
- a synapse S connects G to the output NeuronGroup H. 
- the variable I of H is a summed variable, but the value summed in each synapse is w(i,j)*v(i,t-d(i,j)), with:
    => i = the synapse's input neuron, j = the synapse's output neuron 
    => w(i,j) = the connection strength between neurons i & j
    => v(i,t-d(i,j)) = the variable v of the input neuron i at time t-d(i,j), with:
        => t  = the current time step of the simulation
        => d(i,j) = delay between i & j. The delays are multiples of defaultclock.dt 
"""

# Adapted from https://gist.github.com/mstimberg/30c64429503400de40e7

def Simulation(nb_neurons, simulation_time):
    print("======  simulation ====== ")
    defaultclock.dt = 1*ms
    start = time.time()
    print(">>> number of neurons: "+str(nb_neurons)+", simulation time: "+str(simulation_time)+", time resolution: "+str(defaultclock.dt))

    #===============> 1)  prepare the synapse delays
    delay_max= 10
    delays = [randint(1,delay_max) for i in range(nb_neurons**2)]


    # ===============> 2)  prepare the Input group & the buffer to keep variable v at previous time steps (up to the highest delay)
    G = NeuronGroup(nb_neurons, '''dv/dt = 0.4/ms : 1               
                           buffer_pointer : integer (shared)
                       ''', method='euler',   threshold='v > 4.0', reset='v = 0.0')

    buffer_size = 1+ max(delays)
    G.variables.add_array('v_buffer', size=(buffer_size, len(G)))
    update_code = '''buffer_pointer = (buffer_pointer + 1) % buffer_size
                     return_ = update_v_buffer(v, v_buffer, buffer_pointer)'''
    buffer_updater = G.run_regularly(update_code, codeobj_class=NumpyCodeObject)

    @implementation('numpy', discard_units=True)
    @check_units(v=1, v_buffer=1, buffer_pointer=1, result = 1) # the units
    def update_v_buffer(v, v_buffer, buffer_pointer):
        '''
        A buffer updated at each time step keeps v(t) for all input neurons of G for a temporal time window defined by the highest delay available
        :param v:
        :param v_buffer:
        :param buffer_pointer:
        :return:
        '''
        v_buffer[buffer_pointer, :] = v
        return 0.0


    # ===============> 3)  prepare the Output group
    H = NeuronGroup(nb_neurons, '''I : 1
                           ''', method='euler')


    # ===============> 4)  prepare the Synapses & how to fetch delayed values of v from input group
    S = Synapses(G,H, '''
    w : 1
    v_delayed : 1 
    delay_step : integer
    I_post = v_delayed*w : 1 (summed)
    ''', method='euler')
    S.connect()
    S.delay_step = delays

    S.variables.add_reference('v_buffer_from_synapse', G, varname='v_buffer', index=None)
    S.variables.add_reference('buffer_pointer_from_synapse', G, varname='buffer_pointer', index=None)
    update_code_syn ='''
                     v_delayed = get_v_delayed(v_buffer_from_synapse, buffer_pointer_from_synapse, buffer_size, i, delay_step)
                     '''
    get_new_v_delayed = S.run_regularly(update_code_syn, codeobj_class=NumpyCodeObject)

    # We write the current values at the row given by the current value of buffer_pointer
    # (which is incremented every time step) and retrieve the delay_steps before this
    # row (using a modulo operator to wrap around at the end of the array). Unfortunately, C++ code generation currently
    # cannot deal with 2d arrays, therefore we can only do this in Python
    @implementation('numpy', discard_units=True)
    @check_units(v_buffer_from_synapse=1, buffer_pointer_from_synapse=1, buffer_size=1, i=1, delay_step=1, result=1)
    def get_v_delayed(v_buffer_from_synapse, buffer_pointer_from_synapse, buffer_size, i, delay_step):
        return v_buffer_from_synapse[(buffer_pointer_from_synapse - delay_step) % buffer_size, i]


    # ===============> 5)  Run the simulation
    build_time = time.time()
    run(simulation_time, report_period=1 * second, profile=True)

    end = time.time()
    print("\n => Total elapsed time: ", datetime.timedelta(seconds=end - start))
    print(" ==> DETAILS: network build time: ", datetime.timedelta(seconds=build_time - start),
          " ; simulation time : ", datetime.timedelta(seconds=end - build_time))
    print(profiling_summary(show=5))


if __name__ == '__main__':
    nb_neurons, simulation_time = 600, 10000*ms
    Simulation(nb_neurons, simulation_time)