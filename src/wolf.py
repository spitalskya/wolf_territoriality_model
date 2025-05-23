import random
from typing import Callable, Literal


class Wolf:
    """Class representing one wolf in the model.

    Attributes:
        tag (str): unique tag of the wolf
        
        x (float): location of the wolf
        dx (float): length of one random walk step
        den (float): location of the wolf's den
        
        q (Literal[0, 1]): value that determines rightward (0) movement of the wolf 
            during active phase or leftward (1)
        
        phases_weight_function (Callable[[float], float]): weighing function of the step 
            probability components
        T (float): period of the `phases_weight_function`
        away_bias_function (Callable[[float], float]): preference of the wolf to move through 
            the territory, utilized in the active movement component
        den_attraction (Callable[[float], float]): den attraction force, outputs from [-1, 1]
        
        interaction_level (float): current intensity of olfactory interaction with other wolves,
            in [0,1]
        interaction_transform_function (Callable[[float], float]): transformation of concentration
            of foreign marks to [0, 1]
        marking_amount_function (Callable[[Wolf], float]): determines "amount" of the scent particles
            left during marking
        
        den_movement_process (Callable[[Wolf], None]): function determining if the wolf's den moves
            and where it moves
    """
    
    tag: str
    
    x: float
    dx: float
    den: float

    q: Literal[0, 1]
    
    phases_weight_function: Callable[[float], float]    
    T: float
    away_bias_function: Callable[[float], float]   
    den_attraction: Callable[[float], float]
            
    interaction_level: float        
    interaction_transform_function: Callable[[float], float]
    marking_amount_function: Callable[["Wolf"], float]
    
    den_movement_process: Callable[["Wolf"], None]
        
    
    def __init__(self, tag: str, starting_location: float, dx: float,
                 q: Literal[0, 1],
                 phases_weight_function: Callable[[float], float], T: float, 
                 away_bias_function: Callable[[float], float],
                 den_attraction: Callable[[int], float],
                 interaction_transform_function: Callable[[float], float],
                 marking_amount_function: Callable[["Wolf"], float],     
                 den_movement_process: Callable[[int, "Wolf"], None],
                 ) -> None:
        """Initializes one instance of wolf in the model.

        Args:
            tag (str): unique tag of the wolf
            starting_location (float): location of the wolf for time = 0,
                also location of its den
            dx (float): length of one random walk step
            q (Literal[0, 1]): value that determines rightward (0) movement of the wolf 
                during active phase or leftward (1)
            phases_weight_function (Callable[[float], float]): weighing function of the step 
                probability components
            T (float): period of the `phases_weight_function`
            away_bias_function (Callable[[float], float]): preference of the wolf to move through 
                the territory, utilized in the active movement component
            den_attraction (Callable[[float], float]): den attraction force, outputs from [-1, 1]
            interaction_transform_function (Callable[[float], float]): transformation of concentration
                of foreign marks to [0, 1]
            marking_amount_function (Callable[[Wolf], float]): determines "amount" of the scent particles
                left during marking
            den_movement_process (Callable[[Wolf], None]): function determining if the wolf's den moves
                and where it moves
        """       
        self.tag = tag
        
        self.x = self.den = starting_location
        self.dx = dx
        self.q = q
        
        self.T = T
        self.away_bias_function = away_bias_function
        self.phases_weight_function = phases_weight_function
        self.den_attraction = den_attraction
                
        self.interaction_level = 0      # no interaction at time = 0
        self.interaction_transform_function = interaction_transform_function
        self.marking_amount_function = marking_amount_function
        
        self.den_movement_process = den_movement_process
    

    def step(self, t: float) -> None:
        """Performs one random walk step with bias dependent on passed time

        Args:
            t (float): time
        """
        self.x += random.choices(
            population=[-self.dx, self.dx], 
            weights=[self.L(t), self.R(t)], 
            k=1
            )[0]
        
        # if somehow stepped outside the boundary, enforce it
        self.x = max(0, min(1, self.x))
                

    def interaction(self, scent_concentration: float) -> None:
        """Sets the interaction level from foreign scent concentration at the wolf's location.

        Args:
            scent_concentration (float): foreign scent concentration at the wolf's location
        """
        self.interaction_level = self.interaction_transform_function(scent_concentration)

               
    def R(self, t: float) -> float:
        """Calculates probability of a random walk step to the right at time t.

        Args:
            t (float): time

        Returns:
            float: probability of a step to the right
        """
        x, q, i = self.x, self.q, self.interaction_level
        v, w = self.away_bias_function, self.phases_weight_function

        # region ([0, 1]) boundary behavior
        if x == 0:
            return 1
        if x == 1:
            return 0
        
        # behavior outside of the boundary
        return (
            0.5*(q*(1 - v(t) + i) + (1 - q)*(1 + v(t) - i)) * w(t)                           # active component
            + (1/2 - 0.25*self.den_attraction(x - self.den) - 0.25*(1-2*q)*i) * (1 - w(t))  # inactive component
            )   

    
    def L(self, t: float) -> float:
        """Calculates probability of a random walk step to the left at time t.

        Args:
            t (float): time

        Returns:
            float: probability of a step to the left
        """
        x, q, i = self.x, self.q, self.interaction_level
        v, w= self.away_bias_function, self.phases_weight_function
        
        # region ([0, 1]) boundary behavior
        if x == 0:
            return 0
        if x == 1:
            return 1
        
        # behavior outside of the boundary
        return (
            0.5*(q*(1 + v(t) - i) + (1 - q)*(1 - v(t) + i)) * w(t)                             # active component
            + (0.5 + 0.25*self.den_attraction(x - self.den) + 0.25*(1 - 2*q)*i) * (1 - w(t))  # inactive component
            )                            
