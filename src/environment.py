import torch
from torch import Tensor
from torch.distributions.uniform import Uniform


class Environment:

    _actor_number: int
    _device: torch.device

    _observation_shape: tuple[int, ...] = (2,)
    _action_shape: tuple[int, ...] = (1,)

    _g: float = 9.807

    _m_p: float = 0.1
    _m_c: float = 1.0
    _l: float = 0.5

    _dt: float = 0.02

    _m: float
    _m_p_l: float

    _start_state_distribution: Uniform

    _max_state: Tensor
    _min_state: Tensor

    _Q: Tensor
    _R: Tensor

    _states: Tensor

    def __init__(
            self,
            actor_number: int,
            device: torch.device
            ) -> None:
        self._actor_number = actor_number
        self._device = device

        self._m = self._m_p + self._m_c
        self._m_p_l = self._m_p*self._l

        max_start_state = torch.tensor(
            [2.0, 0.01, torch.pi/4.0, 0.01],
            device=device
        )
        min_start_state = max_start_state.negative()
        self._start_state_distribution = Uniform(
            min_start_state, max_start_state
        )

        self._max_state = torch.tensor(
            [4.0, 1.0, torch.pi, 1.0],
            device=device
        )
        self._min_state = self._max_state.negative()

        self._Q = torch.tensor([1e0, 1e-3, 1e0, 1e-3], device=device).diag()
        self._R = torch.tensor([1e-6], device=device).diag()

        self._states = self._start_state_distribution \
            .sample(sample_shape=(self._actor_number,))

    def __call__(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        x, x_dot, theta, theta_dot = self._states.split(split_size=1, dim=1)

        cos_theta = theta.cos()
        sin_theta = theta.sin()

        temp = (actions + self._m_p_l*theta_dot.square()*sin_theta)/self._m
        theta_double_dot = (self._g*sin_theta - cos_theta*temp)/(
            self._l*(4.0/3.0 - self._m_p*cos_theta.square()/self._m)
        )
        x_double_dot = temp - self._m_p_l*theta_double_dot*cos_theta/self._m

        x_dot += x_double_dot*self._dt
        x += x_dot*self._dt
        theta_dot += theta_double_dot*self._dt
        theta += theta_dot*self._dt

        theta += torch.floor((torch.pi - theta)/(2.0*torch.pi))*2.0*torch.pi

        self._states = torch.column_stack([
            x, x_dot, theta, theta_dot
        ]).clamp(self._min_state, self._max_state)

        observations = torch.column_stack([x, theta])

        x, u = self._states, actions
        costs = torch.einsum('bi,ij,bj->b', x, self._Q, x) \
            + torch.einsum('bi,ij,bj->b', u, self._R, u)

        rewards = costs.negative().exp().unsqueeze(-1)

        return observations, rewards

    def reset(self) -> Tensor:
        self._states = self._start_state_distribution \
            .sample(sample_shape=(self._actor_number,))

        x, _, theta, _ = self._states.split(split_size=1, dim=1)
        observations = torch.column_stack([x, theta])

        return observations

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._observation_shape

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape