from qLearning.dataLoader import DataLoader, load_env_from_file
from qLearning.environment import Environment
from qLearning.config import *
from qLearning.agent import *
import numpy as np

## main code

def run(env, agent):
  steps = []
  all_costs = []
  for episode in range(num_episodes):
    observation = env.reset() #환경 초기화
    print("Episode: {} / {}".format(episode+1, num_episodes))
    step = 0
    cost = 0

    while True:
      # print("episode:", episode, "step:", step)
      s = env.states.index(observation['state'])
      # print("State: ", observation['state'])

      action = agent._act(s)
      # print("action =", action)
      observation_next, reward, done = env.step(action)

      s_next = env.states.index(observation_next['state'])
      cost += agent._update_q_value(s, action, reward, s_next, eta)

      # Swap observation
      observation = observation_next

      step += 1

      if done:
        steps += [step]
        all_costs += [cost]
        break

  env.display()
  agent.plot_results(steps, all_costs)

if __name__ == "__main__":
    load_data = load_env_from_file(load_path, load_file)
    network_env = load_data['network_env']
    data_lst = load_data['data list']
    ctrl = load_data['controller']
    request_lst = load_data['request list']
    popularity = [s.popularity for s in load_data['controller'].svr_lst]

    env = Environment(network_env=network_env, popularity=popularity, cache_size=network_env['cache size'], num_server=network_env['num server'], num_data=network_env['num data'], reward=reward_params, controller=ctrl, data=data_lst)
    Q = np.zeros([env.n_states, env.n_actions])

    agent = Agent(env, gamma, Q)
    run(env, agent)


    '''
    # make popularity
    p_maker = MakePopularity(num_files, zipf_param)
    popularity = p_maker.get_popularity(num_servers)
    env = Environment(popularity, cache_size, num_servers, num_files, reward_params, env_params, queue_length)

    Q = np.zeros([env.n_states, env.n_actions])
    agent = Agent(env, gamma, Q)

    # for i in range(num_servers):
    #     agent.set_cooperNet(MECServer(i, cache_size))

    # print(env.add_servers(agent.get_cooperNet()))

    run(env)
    '''