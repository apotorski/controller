# Controller
The policy gradient-based algorithm that learns to control a nonlinear unstable underactuated partially observable system.

## Environment
The environment is the inverted pendulum - a pole hinged to a movable cart. The goal is to control a cart's position while balancing a pole, by applying forces to the cart.

### State
The environment's state transitions follow a set of differential equations
```math
\ddot{x} = \frac{a + m_{p}l \left( \dot{\theta}^{2}\sin{\theta} - \ddot{\theta}\cos{\theta} \right)}{m_{c} + m_{p}}
```
```math
\ddot{\theta} = \left( g\sin{\theta} - \cos{\theta} \left( \frac{a + m_{p}l\dot{\theta}^{2}\sin{\theta}}{m_{c} + m_{p}} \right) \right) \left( l \left( \frac{4}{3} - \frac{m_{p}\cos^{2}{\theta}}{m_{c} + m_{p}} \right) \right)^{-1}
```
where $`a`$ is a horizontal force applied to the cart, $`x`$ is a cart horizontal position, $`\theta`$ is an angle between pole and vertical, $`m_{c}`$ is a cart mass, $`m_{p}`$ is a pendulum mass, $`l`$ is half the pole length and $`g`$ is the gravitational acceleration.

Environment's state consists of $`x`$, $`\theta`$ and their derivatives with respect to time measured at time step $`t`$ ($`s_{t} = [x_{t}, \dot{x}_{t}, \theta_{t}, \dot{\theta}_{t}]^{\top}`$). The agent observes $`x`$ and $`\theta`$ ($`o_{t} = [x_{t}, \theta_{t}]^{\top}`$).

### Reward
The reward is a transformed linear-quadratic cost for the system
```math
c_{t} = s_{t + 1}^{\top} Q s_{t + 1} + a_{t}^{\top} R a_{t}
```
```math
r_{t} = \exp(-c_{t})
```
where $`Q`$ is a state cost matrix and $`R`$ is a control cost matrix.

## Agent
The agent is represented by a policy $`\pi_{\theta}(a|z)`$ - a probability distribution of action $`a`$ conditioned on a latent state $`z`$ (which depends on observation $`o`$ and previous latent state) and parametrized by $`\theta`$ trained with the policy gradient method. The action distribution is assumed to be a normal distribution ($`\mathcal{N} \left( a; \mu_{\theta}(z), \sigma_{\theta}^{2}(z) \right)`$). A control signal is the most likely action
```math
\arg\max_{a} \pi_{\theta}(a|z) = \mu_{\theta}(z)
```

### Policy
Policy $`\pi_{\theta}`$ is trained through iterative maximization of the objective that encourages advantageous actions and penalizes divergence between policy iterations
```math
\arg\max_{\theta} \mathbb{E}_{t} \left[ \frac{\pi_{\theta}(a_{t}|z_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|z_{t})} A_{t} - \beta D_{\text{KL}} \left[ \pi_{\theta_{\text{old}}}(\cdot|z_{t})||\pi_{\theta}(\cdot|z_{t}) \right] \right]
```
where $`\beta`$ is a penalty coefficient and $`A_{t}`$ is a $`\gamma\lambda`$-discounted advantage estimate at time step $`t`$ computed with an approximate value function $`V_{\phi}`$
```math
\delta_{t} = r_{t} + \gamma V_{\phi}(z_{t + 1}) - V_{\phi}(z_{t})
```
```math
A_{t} = \sum_{l = 0}^{\infty} \left( \gamma\lambda \right)^{l} \delta_{t + l}
```

### Value function
The approximation of a value function $`V_{\phi}`$ is assumed to be the mean of a normal distribution and is trained through minimization of a negative log-likelihood
```math
\arg\min_{\phi} \mathbb{E}_{t} \left[ -\ln{\mathcal{N} \left( R_{t}; V_{\phi}(z_{t}), 1 \right) } \right] = \arg\min_{\phi} \mathbb{E}_{t} \left[ \left( R_{t} - V_{\phi}(z_{t}) \right)^{2} \right]
```
where $`R_{t}`$ is a $`\gamma`$-discounted return at time step $`t`$
```math
R_{t} = \sum_{l = 0}^{\infty} \gamma^{l}r_{t + l}
```

## Quick start
To install dependencies, run
```bash
pip install -r requirements.txt
```

To train an agent, run
```bash
python src/train_agent.py \
    --actor_number 1024 \
    --episode_number 1 \
    --episode_length 256 \
    --gamma 0.99 \
    --lambda_ 0.97 \
    --beta 1.0 \
    --policy_learning_rate 1e-3 \
    --policy_weight_decay 1e-6 \
    --value_function_learning_rate 1e-3 \
    --value_function_weight_decay 1e-6 \
    --batch_size 64 \
    --epoch_number 16 \
    --iteration_number 250 \
    --agent_save_path models/agent.pt
```

To evaluate an agent, run
```bash
python src/evaluate_agent.py \
    --agent_load_path models/agent.pt \
    --episode_length 256
```

## References
[Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.](https://arxiv.org/abs/1707.06347)

[Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.](https://arxiv.org/abs/1506.02438)

[Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, *9*(8), 1735-1780.](https://www.bioinf.jku.at/publications/older/2604.pdf)

[Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system. *Center for Cognitive and Neural Studies (Coneural)*, *Romania*, *63*.](https://coneural.org/florian/papers/05_cart_pole.pdf)