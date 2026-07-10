# %%
import numpy as np
import math
import matplotlib.pyplot as plt
class TrajectoryEnv:
    def __init__(self, max_dx=0.3):
        self.max_steps = 100
        self.max_dx = max_dx
        self.reset()

    def _get_state(self):
        onde_y = np.sin(self.x)
        onde_pente = np.cos(self.x)
        erreur_relative = self.y - onde_y
        return np.array([onde_y, onde_pente, erreur_relative], dtype=np.float32)

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.current_step = 0

        self.history_x = [self.x]
        self.history_y = [self.y]

        return self._get_state()

    def step(self, action):
        self.current_step += 1
        
        dx = np.clip(action[0], 0.0, 1.0)*self.max_dx
        dy = np.clip(action[1], -1.0, 1.0)
        
        self.x += dx
        self.y += dy

        self.history_x.append(self.x)
        self.history_y.append(self.y)
        
        cible_y = np.sin(self.x)
        erreur = abs(self.y - cible_y)

        k = 4.0
        reward = dx * np.exp(-k * erreur ** 2)
        
        done = self.current_step >= self.max_steps
        
        if erreur > 2.0:
            reward -= 5.0
            done = True
            
        return self._get_state(), float(reward), done
    
    def render(self):
        x_parfait = np.linspace(0, max(1.0, self.x), 2000)
        y_parfait = np.sin(x_parfait)

        plt.figure(figsize=(10, 5))
        plt.plot(x_parfait, y_parfait, "g--", label="Trajectoire ideale y = sin(x)")
        plt.plot(self.history_x, self.history_y, "b-o", markersize=3, label="Trajectoire de l'IA")
        plt.title(f"Episode termine. Position finale : x={self.x:.1f}")
        plt.legend()
        plt.grid(True)
        plt.show



# %%
env = TrajectoryEnv()
state = env.reset()
done = False
score_total = 0

while not done:
    rand_act = np.random.uniform(low=-1.0, high=1.0, size=2)
    state, reward, done = env.step(rand_act)
    score_total += reward

print(f"Score final de l'agent aléatoire : {score_total:.2f}")
env.render()

# %%
import torch
import torch.nn as nn
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(self, state_dim=2, action_dim=2):
        super(PPOAgent, self).__init__()

        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mean = self.actor_mean(state_tensor)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(axis=-1)
        state_value = self.critic(state_tensor)
        return action.flatten().numpy(), action_logprob.item(), state_value.item()

# %%
agent = PPOAgent()
etat_test = np.array([1.5, 0.5])
action_choisie, logprob, note_du_coach = agent.act(etat_test)

print("État observé :", etat_test)
print("Action décidée (dx, dy) :", action_choisie)
print("Probabilité de ce choix (log) :", logprob)
print("Note donnée par le Critique :", note_du_coach)

# %%
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

# %%
import torch.optim as optim

optimizer_actor = optim.Adam(agent.parameters(), lr = 0.0003)
optimizer_critic = optim.Adam(agent.critic.parameters(), lr = 0.001)

def ppo_update(agent, memory, gamma=0.99, eps_clip=0.2, k_epochs=4):
    returns = []
    discounted_reward = 0

    for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
        if done:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)

    old_states = torch.FloatTensor(np.array(memory.states))
    old_actions = torch.FloatTensor(np.array(memory.actions))
    old_logprobs = torch.FloatTensor(memory.logprobs)
    old_values = torch.FloatTensor(memory.values)
    returns = torch.FloatTensor(returns)

    returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    advantages = returns - old_values.detach()

    for _ in range(k_epochs):

        action_mean = agent.actor_mean(old_states)
        action_std = torch.exp(agent.actor_log_std)
        dist = Normal(action_mean, action_std)

        new_logprobs = dist.log_prob(old_actions).sum(axis=-1)

        state_values = agent.critic(old_states).squeeze()

        ratios = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(state_values, returns)

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

# %%
from IPython.display import clear_output
import matplotlib.pyplot as plt

env = TrajectoryEnv()
agent = PPOAgent(state_dim=3, action_dim=2)
memory = PPOMemory()

max_episodes = 500
update_timestep = 2000
timestep_counter = 0

historique_scores = []

print("Début de l'entraînement...")

for episode in range(1, max_episodes + 1):
    state = env.reset()
    score_episode = 0
    done = False
    while not done:
        timestep_counter += 1

        action, logprob, value = agent.act(state)

        next_state, reward, done = env.step(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.rewards.append(reward)
        memory.dones.append(done)

        state = next_state
        score_episode += reward

        if timestep_counter % update_timestep == 0:
            ppo_update(agent, memory)
            memory.clear()

    historique_scores.append(score_episode)

    if episode % 50 == 0:
        clear_output(wait=True)
        score_moyen = np.mean(historique_scores[-50:])
        print(f"Épisode {episode}/{max_episodes} | Score moyen (50 derniers) : {score_moyen:.2f}")
        env.render()

plt.figure(figsize=(10, 5))
plt.plot(historique_scores, color='purple')
plt.title("Courbe d'apprentissage : Évolution du Score")
plt.xlabel("Épisodes")
plt.ylabel("Score Total")
plt.grid(True)
plt.show()




