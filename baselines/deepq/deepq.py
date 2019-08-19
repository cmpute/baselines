import os
import os.path as osp
import tempfile
from collections import deque
from rtree import index as rindex

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

######### TOOLS for RL-S #########

def _wrap_data(obs,action,rew,new_obs,done):
    """
    zhong: prepare to save the visited data
    """
    data = np.append(obs[0],action)
    data = np.append(data,float(rew))
    data = np.append(data,new_obs[0])
    data = np.append(data,float(done))
    return data

def _extract_data(data):

    return data[0:14],data[14:15],data[15:16],data[16:30],data[30:31]

def _calculate_visited_times(obs,visited_state_tree):

    return sum(1 for _ in visited_state_tree.intersection(obs.tolist()[0]))

def _calculate_statistics_index(obs,visited_state_value,visited_state_tree,action = -1):
    """
    Calculate 
    """
    if _calculate_visited_times(obs,visited_state_tree) == 0:
        return -1, -1, -1

    value_list = [visited_state_value[idx] for idx in visited_state_tree.intersection(obs.tolist()[0])]
    value_array_av = np.array(value_list)
    value_array = value_array_av[:,1]
    # value_array_rule = value_array[value_array[:,0]==0][:,1]
    # value_array_RL = value_array
    mean = np.mean(value_array)
    var = np.var(value_array)
    sigma = np.sqrt(var)

    return mean,var,sigma

########## TOOLS for RL-S #########


def generate_RLS_action(obs,q_function_cz,action,
                        visited_state_rule_value,visited_state_rule_tree,
                        visited_state_RL_value,visited_state_RL_tree,
                        is_training = True,
                        visited_times_thres = 30,
                        confidence_thres = 0.50,
                        rule_based_safe_thres = 0.9):
    """
    Zhong: generate RLS action
    """
    RLS_action = 0
    visited_times_rule = _calculate_visited_times(obs,visited_state_rule_tree)
    visited_times_RL = _calculate_visited_times(obs,visited_state_RL_tree)

    mean_rule, var_rule, sigma_rule = _calculate_statistics_index(obs,visited_state_rule_value,visited_state_rule_tree)
    mean_RL, var_RL, sigma_RL = _calculate_statistics_index(obs,visited_state_RL_value,visited_state_RL_tree)
    if is_training:
        if visited_times_rule > visited_times_thres and visited_times_rule > visited_times_RL and mean_rule < rule_based_safe_thres:
            RLS_action = action
            return RLS_action
        else:
            RLS_action = 0
            return RLS_action
    else:
        # print("rule:",visited_times_rule,mean_rule,"RL:",visited_times_RL,mean_RL)
        # if obs[0,1] < 5:
        #     return 0
        # else:
        #     return action
        
        if action == 0:
            return action
        
        if visited_times_rule < visited_times_thres or visited_times_RL < 10 or mean_rule > -0.1:
            RLS_action = 0
            return RLS_action
        
        var_diff = var_rule/visited_times_rule + var_RL/visited_times_RL
        sigma_diff = np.sqrt(var_diff)
        mean_diff = mean_RL - mean_rule

        z = mean_diff/sigma_diff
        # print(action,norm.cdf(z))
        if norm.cdf(z)>confidence_thres:
            RLS_action = action
        else:
            RLS_action = 0

        return RLS_action


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    
    ############################## RL-S Prepare #############################################
    
    # model saved name
    saved_name = "0817"

    #####
    # Setup Training Record
    #####
    save_new_data = False
    create_new_file = False
    create_new_file_rule = create_new_file
    save_new_data_rule = save_new_data

    create_new_file_RL = False
    save_new_data_RL = save_new_data
    
    create_new_file_replay_buffer = False
    save_new_data_replay_buffer = save_new_data

    is_training = False
    trajectory_buffer = deque(maxlen=20)

    if create_new_file_replay_buffer:
        if osp.exists("recorded_replay_buffer.txt"):
            os.remove("recorded_replay_buffer.txt")
    else:
        replay_buffer_dataset = np.loadtxt("recorded_replay_buffer.txt")
        for data in replay_buffer_dataset:
            obs, action, rew, new_obs, done = _extract_data(data)
            replay_buffer.add(obs, action, rew, new_obs, done)

    recorded_replay_buffer_outfile = open("recorded_replay_buffer.txt","a")
    recorded_replay_buffer_format = " ".join(("%f",)*31)+"\n"
    
    #####
    # Setup Rule-based Record
    #####
    create_new_file_rule = True

    # create state database
    if create_new_file_rule:
        if osp.exists("state_index_rule.dat"):
            os.remove("state_index_rule.dat")
            os.remove("state_index_rule.idx")
        if osp.exists("visited_state_rule.txt"):
            os.remove("visited_state_rule.txt")
        if osp.exists("visited_value_rule.txt"):
            os.remove("visited_value_rule.txt")

        visited_state_rule_value = []
        visited_state_rule_counter = 0
    else:
        visited_state_rule_value = np.loadtxt("visited_value_rule.txt")
        visited_state_rule_value = visited_state_rule_value.tolist()
        visited_state_rule_counter = len(visited_state_rule_value)

    visited_state_rule_outfile = open("visited_state_rule.txt", "a")
    visited_state_format = " ".join(("%f",)*14)+"\n"

    visited_value_rule_outfile = open("visited_value_rule.txt", "a")
    visited_value_format = " ".join(("%f",)*2)+"\n"

    visited_state_tree_prop = rindex.Property()
    visited_state_tree_prop.dimension = 14
    visited_state_dist = np.array([[0.2, 2, 10, 0.2, 2, 10, 0.2, 2, 10, 0.2, 2, 10, 0.2, 2]])
    visited_state_rule_tree = rindex.Index('state_index_rule',properties=visited_state_tree_prop)

    #####
    # Setup RL-based Record
    #####

    if create_new_file_RL:
        if osp.exists("state_index_RL.dat"):
            os.remove("state_index_RL.dat")
            os.remove("state_index_RL.idx")
        if osp.exists("visited_state_RL.txt"):
            os.remove("visited_state_RL.txt")
        if osp.exists("visited_value_RL.txt"):
            os.remove("visited_value_RL.txt")

    if create_new_file_RL:
        visited_state_RL_value = []
        visited_state_RL_counter = 0
    else:
        visited_state_RL_value = np.loadtxt("visited_value_RL.txt")
        visited_state_RL_value = visited_state_RL_value.tolist()
        visited_state_RL_counter = len(visited_state_RL_value)

    visited_state_RL_outfile = open("visited_state_RL.txt", "a")
    visited_state_format = " ".join(("%f",)*14)+"\n"

    visited_value_RL_outfile = open("visited_value_RL.txt", "a")
    visited_value_format = " ".join(("%f",)*2)+"\n"

    visited_state_tree_prop = rindex.Property()
    visited_state_tree_prop.dimension = 14
    visited_state_dist = np.array([[0.2, 2, 10, 0.2, 2, 10, 0.2, 2, 10, 0.2, 2, 10, 0.2, 2]])
    visited_state_RL_tree = rindex.Index('state_index_RL',properties=visited_state_tree_prop)


    ############################## RL-S Prepare End #############################################
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))


        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action, q_function_cz = act(np.array(obs)[None], update_eps=update_eps, **kwargs)
            
            # RLS_action = generate_RLS_action(obs,q_function_cz,action,visited_state_rule_value,
            #                                 visited_state_rule_tree,visited_state_RL_value,
            #                                 visited_state_RL_tree,is_training)

            RLS_action = 0

            env_action = RLS_action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)

            ########### Record data in trajectory buffer and local file, but not in replay buffer ###########

            trajectory_buffer.append((obs, action, float(rew), new_obs, float(done)))

            # Store transition in the replay buffer.
            # replay_buffer.add(obs, action, rew, new_obs, float(done))

            obs = new_obs
            episode_rewards[-1] += rew # safe driving is 1, collision is 0


            while len(trajectory_buffer)>10:
                # if safe driving for 10(can be changed) steps, the state is regarded as safe
                obs_left, action_left, rew_left, new_obs_left, done_left = trajectory_buffer.popleft()
                # save this state in local replay buffer file
                if save_new_data_replay_buffer:
                    recorded_data = _wrap_data(obs_left, action_left, rew_left, new_obs_left, done_left)
                    recorded_replay_buffer_outfile.write(recorded_replay_buffer_format % tuple(recorded_data))
                # put this state in replay buffer
                replay_buffer.add(obs_left[0], action_left, float(rew_left), new_obs_left[0], float(done_left))
                action_to_record = action_left
                r_to_record = rew_left
                obs_to_record = obs_left

                # save this state in rule-based or RL-based visited state
                if action_left == 0:
                    if save_new_data_rule:
                        visited_state_rule_value.append([action_to_record,r_to_record])
                        visited_state_rule_tree.insert(visited_state_rule_counter,
                            tuple((obs_to_record-visited_state_dist).tolist()[0]+(obs_to_record+visited_state_dist).tolist()[0]))
                        visited_state_rule_outfile.write(visited_state_format % tuple(obs_to_record[0]))
                        visited_value_rule_outfile.write(visited_value_format % tuple([action_to_record,r_to_record]))
                        visited_state_rule_counter += 1
                else:
                    if save_new_data_RL:
                        visited_state_RL_value.append([action_to_record,r_to_record])
                        visited_state_RL_tree.insert(visited_state_RL_counter,
                            tuple((obs_to_record-visited_state_dist).tolist()[0]+(obs_to_record+visited_state_dist).tolist()[0]))
                        visited_state_RL_outfile.write(visited_state_format % tuple(obs_to_record[0]))
                        visited_value_RL_outfile.write(visited_value_format % tuple([action_to_record,r_to_record]))
                        visited_state_RL_counter += 1

            ################# Record data end ########################
            
            
            if done:
                """ 
                Get collision or out of multilane map
                """
                ####### Record the trajectory data and add data in replay buffer #########
                _, _, rew_right, _, _ = trajectory_buffer[-1]

                while len(trajectory_buffer)>0:
                    obs_left, action_left, rew_left, new_obs_left, done_left = trajectory_buffer.popleft()
                    action_to_record = action_left
                    r_to_record = (rew_right-rew_left)*gamma**len(trajectory_buffer) + rew_left
                    # record in local replay buffer file
                    if save_new_data_replay_buffer:
                        obs_to_record = obs_left
                        recorded_data = _wrap_data(obs_left, action_left, r_to_record, new_obs_left, done_left)
                        recorded_replay_buffer_outfile.write(recorded_replay_buffer_format % tuple(recorded_data))
                    # record in replay buffer for trainning
                    replay_buffer.add(obs_left[0], action_left, float(r_to_record), new_obs_left[0], float(done_left))

                    # save visited rule/RL state data in local file
                    if action_left == 0:
                        if save_new_data_rule:
                            visited_state_rule_value.append([action_to_record,r_to_record])
                            visited_state_rule_tree.insert(visited_state_rule_counter,
                                tuple((obs_to_record-visited_state_dist).tolist()[0]+(obs_to_record+visited_state_dist).tolist()[0]))
                            visited_state_rule_outfile.write(visited_state_format % tuple(obs_to_record[0]))
                            visited_value_rule_outfile.write(visited_value_format % tuple([action_to_record,r_to_record]))
                            visited_state_rule_counter += 1
                    else:
                        if save_new_data_RL:
                            visited_state_RL_value.append([action_to_record,r_to_record])
                            visited_state_RL_tree.insert(visited_state_RL_counter,
                                tuple((obs_to_record-visited_state_dist).tolist()[0]+(obs_to_record+visited_state_dist).tolist()[0]))
                            visited_state_RL_outfile.write(visited_state_format % tuple(obs_to_record[0]))
                            visited_value_RL_outfile.write(visited_value_format % tuple([action_to_record,r_to_record]))
                            visited_state_RL_counter += 1

                ####### Recorded #####

                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            ############### Trainning Part Start #####################
            if not is_training:
                # don't need to train the model
                continue

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

                    rew_str = str(mean_100ep_reward)
                    path = osp.expanduser("~/models/carlaok_checkpoint/"+saved_name+"_"+rew_str)
                    act.save(path)

        #### close the file ####
        visited_state_rule_outfile.close()
        visited_value_rule_outfile.close()
        recorded_replay_buffer_outfile.close()
        if not is_training:
            testing_record_outfile.close()
        #### close the file ###

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
