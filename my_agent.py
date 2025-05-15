import numpy as np
import pygame
import torch
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import random
from collections import deque

STUDENT_ID = 'a1884774'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim=3, output_dim=2, learning_rate=1e-3)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=3, output_dim=2, learning_rate=1e-3)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  # probability ε in Algorithm 2
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.n = 32  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2

        self.target_update_freq = 20  # every N updates, sync the target network
        self.update_counter = 0


        self.prev_state = None
        self.prev_action = None

        if self.mode == 'eval':
            self.epsilon = 0.0

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)
    
    def BUILD_STATE(self, state: dict) -> np.ndarray:
        bird_y = state['bird_y'] / state['screen_height']
        bird_velocity = state['bird_velocity'] / 10.0  # assuming range [-10, 10]

        if state['pipes']:
            next_pipe = state['pipes'][0]
            pipe_x = next_pipe['x']
            pipe_top = next_pipe['top']
            pipe_bottom = next_pipe['bottom']
            pipe_center_y = (pipe_top + pipe_bottom) / 2

            horiz_dist = (pipe_x - state['bird_x']) / state['screen_width']
            vert_dist = (state['bird_y'] - pipe_center_y) / state['screen_height']
        else:
            horiz_dist = 1.0  # assume pipe far to the right
            vert_dist = 0.0   # neutral vertical offset

        return np.array([
            bird_velocity,
            horiz_dist,
            vert_dist
        ], dtype=np.float32)




    
    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """

        state_vec = self.BUILD_STATE(state)

        # ε-greedy: explore or exploit
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            # random action (explore)
            a_t = np.random.choice(list(action_table.values())[:2])  # avoid 'quit_game'
        else:
            # predict Q-values and choose best action (exploit)
            q_values = self.network.forward(torch.from_numpy(state_vec.reshape(1, -1)).float())
            q_values_np = q_values.detach().numpy()  # detach from computation graph
            a_t = int(np.argmax(q_values_np))


        self.prev_state = state
        self.prev_action = a_t

        return a_t


    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.prev_state is None or self.prev_action is None:
            return  # skip first step

        s = self.BUILD_STATE(self.prev_state)
        a = self.prev_action
        s_next = self.BUILD_STATE(state)

        done = state['done']
        done_type = state['done_type']
        r = self.REWARD(state, done_type)

        # store (s, a, r, s') in memory
        self.storage.append((s, a, r, s_next, done))

        # only train in training mode and if we have enough data
        if self.mode != 'train' or len(self.storage) < self.n:
            return

        # sample a minibatch
        batch = random.sample(self.storage, k=self.n)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.from_numpy(np.array(states)).float()
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        # Compute target Q-values
        with torch.no_grad():
            q_next = self.network2(next_states_tensor)
            q_next_max = torch.max(q_next, dim=1)[0]
            targets = rewards_tensor + (1 - dones_tensor) * self.discount_factor * q_next_max

        # Predicted Q-values
        q_values = self.network(states_tensor)
        q_pred = q_values.gather(1, actions_tensor.view(-1, 1)).squeeze()

        loss = torch.nn.functional.mse_loss(q_pred, targets)

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()


        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        # forget old state/action
        self.prev_state = None
        self.prev_action = None

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def REWARD(self, state: dict, done_type: str) -> float:
        if done_type == 'not_done':
            return 0.1  # small reward for staying alive
        elif done_type == 'hit_pipe':
            return -1.0
        elif done_type == 'off_screen':
            return -5.0  # slightly worse
        elif done_type == 'well_done':
            return 10.0  # optional for capped runs
        return -0.1

        

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    # env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=10)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=1)
    agent = MyAgent(show_screen=False)

    episodes = 10000

    best_score = 0
    best_mileage = 0

    for episode in range(episodes):
        env.play(player=agent)

        print(f"Episode {episode} — Score: {env.score}, Mileage: {env.mileage}")

        # Save best model by score or mileage
        if env.score > best_score or (env.score == best_score and env.mileage > best_mileage):
            best_score = env.score
            best_mileage = env.mileage
            agent.save_model(path='my_model.ckpt')
            print(f"✅ Saved new best model — Score: {best_score}, Mileage: {best_mileage}")


    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
