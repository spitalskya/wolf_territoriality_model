import numpy as np
from src.wolf import Wolf


class ScentMarks:
    """Class representing the scent marking functions of the wolves

    Attributes:
        scent_marks (np.ndarray): matrix where rows are scent marking functions for each wolf,
            columns represent locations, stored only for current iteration
        dt (float): time increment
        d (float): diffusion coefficient
        M (int): spatial discretization
        wolf_tag_to_index (dict[str, int]): map from wolf tag to row index in `self.scent_marks`
    """
    
    scent_marks: np.ndarray              
    dt: float           
    d: float         
    M: int
    wolf_tag_to_index: dict[str, int]   
    
    
    def __init__(self, dt: float, diffusion_coefficient: float, M: int, wolves: list[Wolf]) -> None:
        """Initializes representation of scent marking function.

        Args:
            dt (float): time increment
            diffusion_coefficient (float): diffusion coefficient
            wolves (list[Wolf]): list of wolves that perform scent marking
            M (int): spatial discretization
        """
        self.scent_marks = np.zeros((len(wolves), M + 1))
        self.dt = dt
        self.d = diffusion_coefficient
        self.M = M
        self.wolf_tag_to_index = {w.tag: i for i, w in enumerate(wolves)}

        
    def update(self, wolves: list[Wolf]) -> None:
        """Updates the scent marking functions
        
        Args:
            wolves (list[Wolf]): wolf that should mark
        """
        updated_marks: np.ndarray = np.copy(self.scent_marks)
        
        # diffusion terms
        left: np.ndarray = np.roll(self.scent_marks, -1, axis=1)
        left[:, -1] = 0
        middle = -2*self.scent_marks
        right: np.ndarray = np.roll(self.scent_marks, 1, axis=1)
        right[:, 0] = 0
        laplacian: np.ndarray = (left + middle + right) # / dx**2
        
        # apply diffusion and decay
        updated_marks += (
            self.d * laplacian              # diffusion
            - self.dt * self.scent_marks    # decay
            ) 
        
        # new marks
        for wolf in wolves:
            row_idx: int = self.wolf_tag_to_index[wolf.tag]
            updated_marks[row_idx, self.location_to_column_index(wolf.x)] += wolf.marking_amount_function(wolf)
        
        # store the new values
        self.scent_marks = updated_marks

        
    def get_foreign_scent_concentration(self, wolf: Wolf) -> float:
        """Gets concentration of foreign scent marks at the wolf's location

        Args:
            wolf (Wolf): wolf whose scent marks should be ignored

        Returns:
            float: concentration of foreign scent marks at the wolf's location
        """
        return (np.sum(
            self.scent_marks[:, self.location_to_column_index(wolf.x)])
            - self.scent_marks[self.wolf_tag_to_index[wolf.tag], self.location_to_column_index(wolf.x)]
            )


    def get_scent_field(self, wolf: Wolf) -> np.ndarray:
        """Get scent marking function of the `wolf` in current iteration

        Args:
            wolf (Wolf): wolf whose scent marking function to return

        Returns:
            np.ndarray: scent marking function of the `wolf`
        """
        return self.scent_marks[self.wolf_tag_to_index[wolf.tag], :]
    

    def location_to_column_index(self, location: float) -> int:
        """Converts (wolf) location to corresponding column index in `self.scent_marks`.

        Args:
            location (float): location to find corresponding column index to

        Returns:
            int: corresponding column index to the passed location
        """
        return round(location*self.M)
