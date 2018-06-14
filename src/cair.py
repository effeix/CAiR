from collections import deque
import numpy as np
from random import sample

import gym
from keras import layers, models, optimizers

IMAGE_SIZE = 16
BATCH_SIZE = 32


class Memory:
    def __init__(self, size, data_type):

        assert size is not None
        assert data_type is not None

        self.memory = deque(maxlen=size)
        self.data_type = data_type

    def save(self, content):
        assert isinstance(content, self.data_type)
        self.memory.append(content)

    def load(self):
        return self.memory

    def erase(self):
        self.memory.clear()


class Agent():
    def __init__(self, input_shape, env):
        self.env = env

        self.__hparams = {
            "EPSILON": 1.0,          # Exploration factor
            "EPSILON_DECAY": 0.99,   # Exploration factor discount
            "EPSILON_LIMIT": 0.01,   # Exploration factor min value
            "GAMMA": 0.95,           # Discount Rate for Bellman's equation
            "LEARNING_RATE": 0.001,  # Learning Rate for DNN optimizer
            "TAU": 0.125             # Target model training interval
        }

        self.__memory = Memory(4000, tuple)

        self.__input_shape = input_shape
        self.__model = self.__build_model()
        self.__target_model = self.__build_model()
        print(self.__model.summary())

    def choose_action(self, state):

        # Apply discount to exploration rate
        self.__hparams["EPSILON"] *= self.__hparams["EPSILON_DECAY"]

        # If exploration rate reached or exceeded limit, set it to min
        self.__hparams["EPSILON"] = max(
            self.__hparams["EPSILON_LIMIT"], self.__hparams["EPSILON"]
        )

        # Each time, it gets harder to explore and easier to exploit
        # If algorithm is exploring, return a random action
        if np.random.rand() < self.__hparams["EPSILON"]:
            return self.env.action_space.sample()

        # Else, predict Q-values (reward for each action) for current state
        tf_state = np.expand_dims(state, axis=0)
        Q_value = self.__model.predict(tf_state)[0]
        # print("Q_value:", Q_value)

        # Return index of the highest Q-value
        return np.argmax(Q_value)

    def get_memory(self):
        return self.__memory.load()

    def play(self, batch_size):

        if len(self.__memory.load()) < batch_size:
            return

        # Get <batch_size> random memory entries
        batch = sample(self.__memory.load(), batch_size)

        for state, action, reward, new_state, done in batch:

            tf_state = np.expand_dims(state, axis=0)
            tf_new_state = np.expand_dims(new_state, axis=0)

            # print("tf_state", tf_state)
            target = self.__target_model.predict(tf_state)

            if done:
                # Game ended, our target is the current state reward
                target[0][action] = reward
            else:
                # Game still running, our target is the current state reward
                # plus next state reward (Bellman's equation)
                target[0][action] = reward + self.__hparams["GAMMA"] \
                    * np.max(self.__target_model.predict(tf_new_state)[0])

            # print("target", target)
            self.__model.fit(tf_state, target, epochs=1, verbose=1)

    def record(self, content):
        self.__memory.save(content)

    def target_train(self):
        # Copy weights from main model to target model

        model_weights = self.__model.get_weights()
        target_model_weights = self.__target_model.get_weights()
        tau = self.__hparams["TAU"]

        for i in range(len(target_model_weights)):
            target_model_weights[i] = \
                model_weights[i] * tau + target_model_weights[i] * \
                (1 - tau)

        self.__target_model.set_weights(target_model_weights)

    def __build_model(self):
        model = models.Sequential()

        l_conv1 = layers.Conv2D(
            16,
            kernel_size=(2, 2),
            activation="relu",
            input_shape=self.__input_shape,
            data_format="channels_last"
        )

        # l_conv2 = layers.Conv2D(
        #     64,
        #     kernel_size=(3, 3),
        #     activation="relu",
        # )

        # l_conv3 = layers.Conv2D(
        #     64,
        #     kernel_size=(3, 3),
        #     activation="relu"
        # )

        l_flat1 = layers.Flatten()

        l_dens1 = layers.Dense(
            128,
            activation="relu"
        )

        l_dens2 = layers.Dense(
            self.env.action_space.n,
            activation="sigmoid"
        )

        # l_dense1 = layers.Dense(64, input_shape=self.__input_shape)
        # l_dense2 = layers.Dense(64)
        # l_dense3 = layers.Dense(64)
        # l_flatt1 = layers.Flatten()
        # l_dense4 = layers.Dense(4)

        model.add(l_conv1)
        # model.add(l_conv2)
        # model.add(l_conv3)
        model.add(l_flat1)
        model.add(l_dens1)
        model.add(l_dens2)

        lr = self.__hparams["LEARNING_RATE"]
        opt = optimizers.Adam(lr=lr)

        model.compile(loss='mse', optimizer=opt)

        return model

    def load(self, name):
        self.__model.load_weights(name)

    def save(self, name):
        self.__model.save_weights(name)


if __name__ == '__main__':

    EPISODES = 10
    STEPS = BATCH_SIZE * 2
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    env = gym.make("MyEnv-v0")

    agent = Agent(INPUT_SHAPE, env)

    # Play <EPISODES> games
    for ep in range(EPISODES):

        state = env.reset()

        # Each game takes <STEPS> steps
        for st in range(STEPS):

            # Display the game
            # env.render()

            # Predict an action and execute it in the environment
            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)

            # Store experience acquired by the "player"
            # after executing an action
            agent.record((state, action, reward, new_state, done))

            # Train main model based on recorded states
            agent.play(BATCH_SIZE)

            # Update target model
            agent.target_train()

            state = new_state

            if done:
                print("GAME FINISHED")
                break

    agent.save("weights.h5")

    try:
        agent.load("weights.h5")
    except OSError:
        pass

    obs = env.reset()

    STEPS = 400
    for st in range(STEPS):

        env.render()

        action = agent.choose_action(obs)
        new_obs, reward, done, info = env.step(action)

        if done:
            print("DONE")

            break
