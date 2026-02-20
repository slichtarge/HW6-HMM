import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    observe_states = mini_hmm['observation_states']
    hidden_states = mini_hmm["hidden_states"]
    priors = mini_hmm['prior_p']
    transitions = mini_hmm['transition_p']
    emissions = mini_hmm['emission_p']

    my_markov = HiddenMarkovModel(observe_states, hidden_states, priors, transitions, emissions)

    # Test Forward Algorithm
    forward_prob = my_markov.forward(mini_input["observation_state_sequence"])
    print(f"Forward probability: {forward_prob}")
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."
    assert forward_prob > 0, "Forward probability should be greater than 0."
    assert forward_prob <= 1, "Forward probability should be less than or equal to 1."
    assert forward_prob == 0.03506441162109375, "Forward probability does not match expected value."

    # Test Viterbi Algorithm
    viterbi_path = my_markov.viterbi(mini_input["observation_state_sequence"])
    print(f"Viterbi path: {viterbi_path}")
    assert len(viterbi_path) == len(mini_input["observation_state_sequence"]), "Viterbi path length does not match observation sequence length."
    assert all(state in hidden_states for state in viterbi_path), "Viterbi path contains invalid hidden states."
    assert viterbi_path == mini_input["best_hidden_state_sequence"].tolist(), "Viterbi path does not match expected best hidden state sequence."

    #edge case: what happens if we give it a sequence that is impossible to generate?
    original_emissions = my_markov.emission_p.copy()
    my_markov.emission_p = np.zeros_like(original_emissions) 
    
    impossible_prob = my_markov.forward(mini_input["observation_state_sequence"])
    assert impossible_prob == 0.0, "Forward probability of an impossible sequence should be 0.0."

    #edge case: what happens if we give it a sequence of length 1?
    single_obs = np.array([mini_input["observation_state_sequence"][0]])
    single_viterbi = my_markov.viterbi(single_obs)
    
    assert len(single_viterbi) == 1, "Viterbi should handle a single observation."

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    observe_states = full_hmm['observation_states']
    hidden_states = full_hmm["hidden_states"]
    priors = full_hmm['prior_p']
    transitions = full_hmm['transition_p']
    emissions = full_hmm['emission_p']

    my_markov = HiddenMarkovModel(observe_states, hidden_states, priors, transitions, emissions)

    # Test Forward Algorithm
    forward_prob = my_markov.forward(full_input["observation_state_sequence"])
    print(f"Forward probability: {forward_prob}")
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."
    assert forward_prob > 0, "Forward probability should be greater than 0."
    assert forward_prob <= 1, "Forward probability should be less than or equal to 1."
    assert forward_prob == 1.6864513843961343e-11, "Forward probability does not match expected value."

    # Test Viterbi Algorithm
    viterbi_path = my_markov.viterbi(full_input["observation_state_sequence"])
    print(f"Viterbi path: {viterbi_path}")
    assert len(viterbi_path) == len(full_input["observation_state_sequence"]), "Viterbi path length does not match observation sequence length."
    assert all(state in hidden_states for state in viterbi_path), "Viterbi path contains invalid hidden states."
    assert viterbi_path == full_input["best_hidden_state_sequence"].tolist(), "Viterbi path does not match expected best hidden state sequence."

    #edge case: what happens if we give it a sequence that is impossible to generate?
    original_emissions = my_markov.emission_p.copy()
    my_markov.emission_p = np.zeros_like(original_emissions) 
    
    impossible_prob = my_markov.forward(full_input["observation_state_sequence"])
    assert impossible_prob == 0.0, "Forward probability of an impossible sequence should be 0.0."

    #edge case: what happens if we give it a sequence of length 1?
    single_obs = np.array([full_input["observation_state_sequence"][0]])
    single_viterbi = my_markov.viterbi(single_obs)

    assert len(single_viterbi) == 1, "Viterbi should handle a single observation."

