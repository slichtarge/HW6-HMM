import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """
        
        # Step 1. Initialize variables
        n_obs = len(input_observation_states)
        n_states = len(self.hidden_states)
        
        #forward_table[t, j] represents P(O_1...O_t, state_t = j)
        forward_table = np.zeros((n_obs, n_states))

        # Step 2. Calculate probabilities
        
        #base case: initialize at time t=0
        first_obs = input_observation_states[0]
        first_obs_idx = self.observation_states_dict[first_obs]
        
        for s in range(n_states):
            #probability = prior * emission
            forward_table[0, s] = self.prior_p[s] * self.emission_p[s, first_obs_idx]

        #recursive step: iterate through the rest of the sequence
        for t in range(1, n_obs):
            current_obs = input_observation_states[t]
            obs_idx = self.observation_states_dict[current_obs]
            
            for j in range(n_states):
                #sum of (previous forward prob * transition to current state)
                prev_sum = 0
                for i in range(n_states):
                    prev_sum += forward_table[t-1, i] * self.transition_p[i, j]
                
                #multiply by emission probability of the current observation
                forward_table[t, j] = prev_sum * self.emission_p[j, obs_idx]

        # Step 3. Return final probability 
        #the total likelihood is the sum of probabilities of all ending states
        forward_probability = np.sum(forward_table[-1, :])
        
        return float(forward_probability)
    
        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))

        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))
        
        # Step 2. Calculate Probabilities
        pointer_table = np.zeros_like(viterbi_table, dtype=int)

        
        for idx, obs in enumerate(decode_observation_states):

            #ok so snag the prob(obs)
            p_obs_given_states = []
            for h_idx, hidden_state in enumerate(self.hidden_states):
                p_obs_given_state = self.emission_p[h_idx, self.observation_states.tolist().index(obs)]
                p_obs_given_states.append(p_obs_given_state)

            #if we are at the first obs,just calculate the priors and save
            if idx == 0:
                for h_idx, hidden_state in enumerate(self.hidden_states):
                    p_secretly_hidden_state = p_obs_given_states[h_idx] * self.prior_p[h_idx]
                    # print(f"0th P({obs} | {hidden_state} ) = ", p_secretly_hidden_state)
                    viterbi_table[idx, h_idx] = p_secretly_hidden_state
                # print()
                continue
                
            #if not the first obs, calculte the options per state
            p_secretly_state_maxes = []

            #for every state
            for h_idx, hidden_state in enumerate(self.hidden_states):
                p_secretly_hidden_state_options = []

                #calc the options of which state we could have come from
                for h_idx2, _ in enumerate(self.hidden_states):
                    option = viterbi_table[idx-1, h_idx2] * self.transition_p[h_idx2, h_idx]
                    p_secretly_hidden_state_options.append(option)

                #take the max, and save it along with the pointer
                p_secretly_state_max = max(p_secretly_hidden_state_options)
                p_secretly_state_maxes.append(p_secretly_state_max)
                pointer_table[idx, h_idx] = p_secretly_hidden_state_options.index(p_secretly_state_max)

            #calculate the overall p_of_state
            p_of_states = (np.array(p_obs_given_states)*np.array(p_secretly_state_maxes)).tolist()

            #save responsibly
            for h_idx, state in enumerate(self.hidden_states):
                # print(f"{idx}th P({obs} | {state}) = ", p_of_states[h_idx])
                viterbi_table[idx, h_idx] = p_of_states[h_idx]
            
        # Step 3. Traceback 
        #start from most probable state
        best_last_state = np.argmax(viterbi_table[-1, :])

        #trace backwards
        best_path = [best_last_state]
        for idx in range(len(decode_observation_states) - 1, 0, -1):
            best_prev_state = pointer_table[idx, best_path[-1]]
            best_path.append(best_prev_state)

        #reverse and map back to actual state names
        best_path.reverse()
        best_path_named = [str(self.hidden_states[i]) for i in best_path]

        # Step 4. Return best hidden state sequence 
        return best_path_named