import numpy as np
from utils import *   # import replay buffer for experienced replay
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam


def LinearDeepQNetwork(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential()
    # (input_dims, ) allows us to pass in a batch or just a single memory
    model.add(Dense(fc1_dims, input_shape=(input_dims, )))
    model.add(Activation('relu'))
    model.add(Dense(fc2_dims))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                 eps_dec, eps_min, mem_size, fname='dqn_model'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = LinearDeepQNetwork(lr, n_actions, input_dims, 128, 128)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        # add an axis to reshape the input shape
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values).astype(np.int32)

            q_eval = self.q_eval.predict(state)
            q_next = self.q_eval.predict(state_)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            #print("action_indices:", action_indices, "action_values:", action_values)
            q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min


    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
