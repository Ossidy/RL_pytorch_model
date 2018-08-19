from torch.autograd import Variable
import numpy as np
import torch
from submodels import *



# a special module that converts [batch, channel, w, h] to [batch, units]


cuda = torch.cuda.is_available()
# if cuda:
#     torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
#                                                      else torch.FloatTensor)

# cuda = False


class Agent(nn.Module):
    def __init__(self, obs_shape=[3, 64, 64], n_actions=5, n_parallel_games=5, reuse=False, memory_dim=128, model_idx=1):
        
        super(Agent, self).__init__()

        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_parallel_games = n_parallel_games
        self.memory_dim = memory_dim

        assert self.obs_shape[1] == self.obs_shape[2], "Current only accept square-shaped images"
        if model_idx == 0:
            assert self.obs_shape[1] == 64, "Only support images with shape of 64*64 now"

        self.decisionUnit = DecisionNetwork(n_actions, memory_dim)
        self.memoryUnit = LongMemoryLSTM(memory_dim)
        self.perceptionUnit = Encoder(memory_dim, obs_shape[0], obs_shape[1], model_idx=model_idx)
        # self.recoveryUnit = Decoder(memory_dim, obs_shape[0], obs_shape[1])
        self.curiosityUnit = CuriosityNetwork(n_actions, 128, obs_shape[0], obs_shape[1]) # 128 is the feature dimension

        # self.shortMemoryUnit = MemoryLSTM(memory_dim)

        if cuda:
            self.decisionUnit.cuda()
            self.memoryUnit.cuda()
            self.perceptionUnit.cuda()
            # self.recoveryUnit.cuda()
            self.curiosityUnit.forwardDynamics.cuda()
            self.curiosityUnit.inverseDynamics.cuda()
            self.curiosityUnit.perception.cuda()
            # self.shortMemoryUnit.cuda()

    def PR_forward(self, obs_t):

        # observations to dense memory and then recovered to reconstructed observations
        encoded_memory = self.perceptionUnit(obs_t)
        recovered_obs = self.recoveryUnit(encoded_memory)

        return recovered_obs

    def PMR_forward(self, prev_state, obs_t):

        encoded_memory = torch.unsqueeze(self.perceptionUnit(obs_t[0]), dim=0)
        # print(encoded_memory[0].shape)
        recovered_obs = torch.unsqueeze(self.recoveryUnit(encoded_memory[0]), dim=0)
        for i in range(1, obs_t.shape[0]):
            encoded_memory_this = self.perceptionUnit(obs_t[i])
            encoded_memory = torch.cat((encoded_memory, torch.unsqueeze(encoded_memory_this, dim=0)), dim=0)
            recovered_obs = torch.cat((recovered_obs, torch.unsqueeze(self.recoveryUnit(encoded_memory_this), dim=0)), dim=0)
        
        # encoded memory
        hx, cx = prev_state
        hx, cx = self.shortMemoryUnit((hx, cx), encoded_memory)
        new_state = (hx, cx)

        x = hx
        recovered_obs_2 = torch.unsqueeze(self.recoveryUnit(x[0]), dim=0)
        for i in range(1, obs_t.shape[0]):
            recovered_obs_2 = torch.cat((recovered_obs_2, torch.unsqueeze(self.recoveryUnit(x[i]), dim=0)), dim=0)

        return recovered_obs, recovered_obs_2, new_state

    def PMD_forward(self, prev_state, obs_t):

        hx, cx = prev_state

        encoded_memory = self.perceptionUnit(obs_t)
        hx, cx = self.memoryUnit((hx, cx), encoded_memory)

        new_state = (hx, cx)
        x = hx

        logits, state_value = self.decisionUnit(x)

        return new_state, (logits, state_value)


    def step(self, prev_state, obs_t): 
        # i.e. perception_memory_decision 
        if not cuda:
            obs_t = Variable(torch.FloatTensor(np.array(obs_t)))
        else:
            obs_t = Variable(torch.FloatTensor(np.array(obs_t)).cuda())

        (h, c), (l, s) = self.PMD_forward(prev_state, obs_t)
        # (h, c), (l, s) = self.decisionUnit(prev_state, obs_t)

        return (h.detach(), c.detach()), (l.detach(), s.detach())

    def sample_actions(self, agent_outputs):
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        # print(torch.sum(probs, dim=1))
        return torch.multinomial(probs, 1)[:, 0].data.cpu().numpy()


    def fix_gradients(self, unit_name, fix=True):
        unit_names = ['decisionUnit', 'memoryUnit', 'perceptionUnit', 'recoveryUnit', 'shortMemoryUnit']
        units = [self.decisionUnit, self.memoryUnit, self.perceptionUnit] #, self.recoveryUnit]#, self.shortMemoryUnit]

        assert unit_name in unit_names

        selected_unit = units[unit_names.index(unit_name)]

        if fix:
            for p in selected_unit.parameters():
                p.requires_grad = False
        else:
            for p in selected_unit.parameters():
                p.requires_grad = True



import gym
from train import *
from atari_util import PreprocessAtari
def make_env():
    game_id="KungFuMasterDeterministic-v0"
    env = gym.make(game_id)
    env = PreprocessAtari(env, height=64, width=64,
                          crop = lambda img: img[60:-30, 15:],
                          color=True, n_frames=1)
    return env



env = make_env()
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
obs = env.reset()
a = Agent(obs_shape=obs_shape, n_actions=5, n_parallel_games=5)
# a.PR_forward(obs)
print("agent created")
