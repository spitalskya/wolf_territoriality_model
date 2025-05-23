from functools import partial
from math import floor
import random
from typing import Callable
import numpy as np
from src.scent_marks import ScentMarks
from src.wolf import Wolf


def build_instances(den_a: int = 0.25, den_b: int = 0.75,
                    T_a: float = 0.5, T_b: float = 0.5,
                    diffusion_coefficient: float = 0.25
                    ) -> tuple[Wolf, Wolf, ScentMarks]:
    
    ######################### discretization, chi #########################
    M: int = 200
    dx: float = 1 / M
    dt: float = 1 / 1000
    chi: float = 200

    ######################### active phase #########################
    cos_squared: Callable[[float, float], float] = lambda t, T: 1-np.cos((np.pi/T)*t)**2
    square_wave: Callable[[float, float], float] = lambda t, T: 0 if (t/T) % 1 < 0.5 else 1
    triangle_wave: Callable[[float, float], float] = lambda t, T: 2*abs(t/T - floor(t/T + 1/2))
    
    omega_a: Callable[[float], float] = lambda t: triangle_wave(t, T_a)
    rho_a: Callable[[float], float] = lambda t: triangle_wave(t, T_a)
    
    omega_b: Callable[[float], float] = lambda t: triangle_wave(t, T_b) 
    rho_b: Callable[[float], float] = lambda t: triangle_wave(t, T_b)
    
    ######################### inactive phase #########################
    psi_a: Callable[[int], float] = lambda x: np.tanh(chi*x)
    psi_b: Callable[[int], float] = lambda x: np.tanh(chi*x)
       
    ######################### interaction #########################
    phi_a: Callable[[float], float] = lambda x: np.tanh(x)
    phi_b: Callable[[float], float] = lambda x: np.tanh(x)
    
    ######################### marked amount #########################
    s: Callable[[Wolf], float] = lambda w: (
         0.1 * np.tanh(chi * abs(w.x - w.den)) 
         + w.interaction_level
         )
    z: Callable[[Wolf], float] = lambda w: (
         0.1 * np.tanh(chi * abs(w.x - w.den))
         + w.interaction_level
         )
    
    ######################### den movement #########################
    # not used in simulations
    den_movement_process_a: Callable[[int, Wolf], None] = no_den_movement
    den_movement_process_b: Callable[[int, Wolf], None] = no_den_movement
    
    """den_movement_process_a: Callable[[int, RandomWolfWalk], None] = partial(
        compound_bernoulli_den_movement, p=1/(T_a/dt)      # expected moves -> once each period
    )"""
    """den_movement_process_b: Callable[[int, RandomWolfWalk], None] = partial(
        compound_bernoulli_den_movement, p=1/(T_a/dt)
    )"""
    
    ######################### build #########################
    wolf_a: Wolf = Wolf(
        tag="A", starting_location=den_a, q=0, dx=dx,
        phases_weight_function=omega_a, 
        T = T_a,
        away_bias_function=rho_a,
        den_movement_process=den_movement_process_a,
        marking_amount_function=s,
        interaction_transform_function=phi_a,
        den_attraction = psi_a)
    
    wolf_b: Wolf = Wolf(
        tag="B", starting_location=den_b, q=1, dx=dx,
        phases_weight_function=omega_b, 
        T = T_b,
        away_bias_function=rho_b,
        den_movement_process=den_movement_process_b,
        marking_amount_function=z,
        interaction_transform_function=phi_b,
        den_attraction = psi_b)
    
    pheromone: ScentMarks = ScentMarks(
        M = M, wolves=[wolf_a, wolf_b],
        dt=dt, diffusion_coefficient=diffusion_coefficient
        )
        
    return wolf_a, wolf_b, pheromone

    
def no_den_movement(wolf: Wolf) -> None:
    pass


def compound_bernoulli_den_movement(wolf: Wolf, p: float) -> None:
    if random.random() < p:
        wolf.den = max(0, min(1, wolf.den + random.choice([-wolf.dx, wolf.dx])))
