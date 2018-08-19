import gym
from atari_util import PreprocessAtari
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

# 

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)
# cuda = False

def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation and memory
        observation = env.reset()
        prev_memories = agent.memoryUnit.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts = agent.step(prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])

            total_reward += reward
            prev_memories = new_memories
            if done: break
                
        game_rewards.append(total_reward)
    return game_rewards

def crossEntropyLoss_one_hot(input, target):
    _, labels = target.max(dim=1)
#     print(labels)
    return nn.CrossEntropyLoss()(input, labels)

def MSELoss(input, target):
    return nn.MSELoss()(input, target)

def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    if not cuda:
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    else:
        y_tensor = y_tensor.type(torch.cuda.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def intrinsic_rewards_loss(agent, states, actions_one_hot, rollout_length, ita=1000):

    inverse_loss = 0
    forward_loss = 0
    rewards_intrinsic = []
    for t in range(rollout_length):
        # do the curiosity forward

        obs_t = states[:, t]
        obs_next_t = states[:, t+1]
        act_t = actions_one_hot[:, t]

        ### use the same perception
        # feat_t = agent.perceptionUnit(obs_t)
        # feat_next_t = agent.perceptionUnit(obs_next_t)

        # act_hat = agent.curiosityUnit.inverseDynamics(feat_t.detach(), feat_next_t.detach())
        # feat_next_t_hat = agent.curiosityUnit.forwardDynamics(act_t, feat_t.detach())

        # use differetn perceptions
        feat_t = agent.curiosityUnit.perception(obs_t)
        feat_next_t = agent.curiosityUnit.perception(obs_next_t)

        act_hat = agent.curiosityUnit.inverseDynamics(feat_t, feat_next_t)
        feat_next_t_hat = agent.curiosityUnit.forwardDynamics(act_t, feat_t.detach())

        inverse_loss += crossEntropyLoss_one_hot(act_hat, act_t)
        forward_loss += MSELoss(feat_next_t_hat, feat_next_t)
        r_intrin = torch.mean((torch.abs(feat_next_t - feat_next_t_hat)), 1)
        rewards_intrinsic.append(r_intrin.data.cpu().numpy())


    inverse_loss = inverse_loss / float(rollout_length)
    forward_loss = forward_loss / float(rollout_length)
    rewards_intrinsic = np.array(rewards_intrinsic) * ita # 0.5 is ita should be tuned latter


    rewards_intrinsic = Variable(torch.from_numpy(np.transpose(rewards_intrinsic)).cuda())

    return inverse_loss, forward_loss, rewards_intrinsic


def A3C_loss(agent, rewards, state_values, logprobas_for_actions, rollout_length, gamma, rewards_intrinsic=None):
    value_loss = 0
    J_hat = 0
    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t]                                # current rewards
        if rewards_intrinsic is not None:
             r_t += rewards_intrinsic[:, t]      

        V_t = state_values[:, t]                           # current state values
        V_next = state_values[:, t + 1].detach()           # next state values
        logpi_a_s_t = logprobas_for_actions[:, t]          # log-probability of a_t in s_t
       
        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + gamma * cumulative_returns
        
        # Compute temporal difference error (MSE for V(s))
        value_loss += (r_t + gamma * V_next - V_t)**2

        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = G_t - V_t
        advantage = advantage.detach()
      
        # compute policy pseudo-loss aka -J_hat.
        J_hat += logpi_a_s_t * advantage

    J_hat = torch.mean(J_hat)
    value_loss = torch.mean(torch.squeeze(value_loss))



    return J_hat, value_loss

def train_on_rollout(agent, opts, states, actions, rewards, is_not_done, prev_memory_states, gamma = 0.99, curiosity=True):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    """

    n_actions = agent.n_actions
    n_parallel_games = agent.n_parallel_games

    if not cuda:
        states = Variable(torch.FloatTensor(np.array(states)))   # shape: [batch_size, time, c, h, w]
        actions = Variable(torch.IntTensor(np.array(actions)))   # shape: [batch_size, time]
        rewards = Variable(torch.FloatTensor(np.array(rewards))) # shape: [batch_size, time]
        is_not_done = Variable(torch.FloatTensor(is_not_done.astype('float32')))  # shape: [batch_size, time]
        rollout_length = rewards.shape[1] - 1
    else:
        states = Variable(torch.FloatTensor(np.array(states)).cuda())   # shape: [batch_size, time, c, h, w]
        actions = Variable(torch.IntTensor(np.array(actions)).cuda())   # shape: [batch_size, time]
        rewards = Variable(torch.FloatTensor(np.array(rewards)).cuda()) # shape: [batch_size, time]
        is_not_done = Variable(torch.FloatTensor(is_not_done.astype('float32')).cuda())  # shape: [batch_size, time]
        rollout_length = rewards.shape[1] - 1

    # predict logits, probas and log-probas using an agent. 
    memory = [m.detach() for m in prev_memory_states]
    
    logits = [] # append logit sequence here
    state_values = [] # append state values here
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        
        # use agent to comute logits_t and state values_t.
        # append them to logits and state_values array      
        memory, (logits_t, values_t) = agent.PMD_forward(memory, obs_t)
        
        logits.append(logits_t)
        state_values.append(values_t)

    logits = torch.stack(logits, dim=1)
    state_values = torch.squeeze(torch.stack(state_values, dim=1))
    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

        
    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim = -1)
   
    if curiosity:
        inverse_loss, forward_loss, rewards_intrinsic = intrinsic_rewards_loss(agent, states, actions_one_hot, rollout_length, ita=50)
        J_hat, value_loss = A3C_loss(agent, rewards, state_values, logprobas_for_actions, rollout_length, gamma, rewards_intrinsic=rewards_intrinsic)
        loss = -J_hat + value_loss + 1 * inverse_loss + 1 * forward_loss

    else:
        J_hat, value_loss = A3C_loss(agent, rewards, state_values, logprobas_for_actions, rollout_length, gamma, rewards_intrinsic=None)
            #regularize with entropy
        entropy = torch.sum(probas * logprobas, dim=2)
        entropy = torch.mean(entropy)
        loss = -J_hat + value_loss - 0.01 * entropy 

    # Gradient descent step
    for opt in opts:
        opt.zero_grad()
    loss.backward()
    for opt in opts:
        opt.step()
    
    if curiosity:
        return loss.data.cpu().numpy(), forward_loss, inverse_loss, rewards_intrinsic

    return loss.data.cpu().numpy()#, forward_loss, inverse_loss, rewards_intrinsic

def train_on_rollout_(agent, opts, states, actions, rewards, is_not_done, prev_memory_states, gamma = 0.99, curiosity=True):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    def crossEntropyLoss_one_hot(input, target):
        _, labels = target.max(dim=1)
    #     print(labels)
        return nn.CrossEntropyLoss()(input, labels)

    MSEloss = nn.MSELoss()


    # cast everything into a variable
    n_actions = agent.n_actions
    n_parallel_games = agent.n_parallel_games


    if not cuda:
        states = Variable(torch.FloatTensor(np.array(states)))   # shape: [batch_size, time, c, h, w]
        actions = Variable(torch.IntTensor(np.array(actions)))   # shape: [batch_size, time]
        rewards = Variable(torch.FloatTensor(np.array(rewards))) # shape: [batch_size, time]
        is_not_done = Variable(torch.FloatTensor(is_not_done.astype('float32')))  # shape: [batch_size, time]
        rollout_length = rewards.shape[1] - 1
    else:
        states = Variable(torch.FloatTensor(np.array(states)).cuda())   # shape: [batch_size, time, c, h, w]
        actions = Variable(torch.IntTensor(np.array(actions)).cuda())   # shape: [batch_size, time]
        rewards = Variable(torch.FloatTensor(np.array(rewards)).cuda()) # shape: [batch_size, time]
        is_not_done = Variable(torch.FloatTensor(is_not_done.astype('float32')).cuda())  # shape: [batch_size, time]
        rollout_length = rewards.shape[1] - 1

    # predict logits, probas and log-probas using an agent. 
    memory = [m.detach() for m in prev_memory_states]
    
    logits = [] # append logit sequence here
    state_values = [] # append state values here
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        
        # use agent to comute logits_t and state values_t.
        # append them to logits and state_values array      
        memory, (logits_t, values_t) = agent.PMD_forward(memory, obs_t)
        
        logits.append(logits_t)
        state_values.append(values_t)

    logits = torch.stack(logits, dim=1)
    state_values = torch.squeeze(torch.stack(state_values, dim=1))
    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

        
    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim = -1)
   

    inverse_loss = 0
    forward_loss = 0
    rewards_intrinsic = []
    for t in range(rollout_length):
        # do the curiosity forward

        obs_t = states[:, t]
        obs_next_t = states[:, t+1]
        act_t = actions_one_hot[:, t]

        ### use the same perception
        # feat_t = agent.perceptionUnit(obs_t)
        # feat_next_t = agent.perceptionUnit(obs_next_t)

        # act_hat = agent.curiosityUnit.inverseDynamics(feat_t.detach(), feat_next_t.detach())
        # feat_next_t_hat = agent.curiosityUnit.forwardDynamics(act_t, feat_t.detach())

        # use differetn perceptions
        feat_t = agent.curiosityUnit.perception(obs_t)
        feat_next_t = agent.curiosityUnit.perception(obs_next_t)

        act_hat = agent.curiosityUnit.inverseDynamics(feat_t.detach(), feat_next_t.detach())
        feat_next_t_hat = agent.curiosityUnit.forwardDynamics(act_t, feat_t)

        inverse_loss += crossEntropyLoss_one_hot(act_hat, act_t)
        forward_loss += MSEloss(feat_next_t_hat, feat_next_t)
        r_intrin = torch.mean((torch.abs(feat_next_t - feat_next_t_hat)), 1)
        rewards_intrinsic.append(r_intrin.data.cpu().numpy())


    inverse_loss = inverse_loss / float(rollout_length)
    forward_loss = forward_loss / float(rollout_length)
    rewards_intrinsic = np.array(rewards_intrinsic) * 1000 # 0.5 is ita should be tuned latter
    # print(rewards_intrinsic)
    # print(rewards)

    rewards_intrinsic = Variable(torch.from_numpy(np.transpose(rewards_intrinsic)).cuda())

    value_loss = 0
    J_hat = 0
    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t] + rewards_intrinsic[:, t]      # current rewards
        V_t = state_values[:, t]                           # current state values
        V_next = state_values[:, t + 1].detach()           # next state values
        logpi_a_s_t = logprobas_for_actions[:, t]          # log-probability of a_t in s_t
       
        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + gamma * cumulative_returns
        
        # Compute temporal difference error (MSE for V(s))
        value_loss += (r_t + gamma * V_next - V_t)**2

        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = G_t - V_t
        advantage = advantage.detach()
      
        # compute policy pseudo-loss aka -J_hat.
        J_hat += logpi_a_s_t * advantage

 




    J_hat = torch.mean(J_hat)
    value_loss = torch.mean(torch.squeeze(value_loss))

    #regularize with entropy
    entropy = torch.sum(probas * logprobas, dim=2)
    entropy = torch.mean(entropy)
    
    # add-up three loss components and average over time
    # loss = -J_hat + value_loss -0.01 * entropy 
    loss = -J_hat + value_loss - 0.00 * entropy + 1 * inverse_loss + 1 * forward_loss
    
    # Gradient descent step
    for opt in opts:
        opt.zero_grad()
    loss.backward()
    for opt in opts:
        opt.step()
    
    
    return loss.data.cpu().numpy(), forward_loss, inverse_loss, rewards_intrinsic