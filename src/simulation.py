from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Callable, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from src.database_handler import DatabaseHandler
from src.instance_builder import build_instances
from src.scent_marks import ScentMarks
from src.wolf import Wolf

config = tuple[float, float, float]


class Simulation:
    """Class that oversees the simulation process, calls `Wolf` and `ScentMarks` methods in 
    appropriate order and gathers data from simulation.

    Attributes:
        wolf_a (Wolf): first (left) wolf
        wolf_b (Wolf): second (right) wolf
        sm (ScentMarks): scent marking functions
    """
    wolf_a: Wolf
    wolf_b: Wolf
    sm: ScentMarks
    
    def __init__(self, wolf_a: Wolf, wolf_b: Wolf, sm: ScentMarks) -> None:
        """Stores the provided instances to use in the simulation

        Args:
            wolf_a (Wolf): first (left) wolf
            wolf_b (Wolf): second (right) wolf
            sm (ScentMarks): scent marking functions
        """
        self.wolf_a = wolf_a
        self.wolf_b = wolf_b
        self.sm = sm

        
    def simulate(self, time_steps: int, track_border: bool = False, 
                 track_scent_marks: bool = False, track_locations: bool = False) -> None:
        """Simulates the model for `time_steps` iterations of length dt (stored in ScentMarks).
        Also, gathers the desired data

        Args:
            time_steps (int): how many iterations to perform
            track_border (bool, optional): whether to store border location, scent concentration 
                at the border and derivatives of the scent marking function at the border. 
                Defaults to False.
            track_scent_marks (bool, optional): whether to store scent marking functions 
                for each time step. 
                Defaults to False.
            track_locations (bool, optional): whether to store wolf locations in each time step. 
                Defaults to False.
        """
        # setup neccessary fields for desired data to gather
        if track_border:
            self.borders = []
            self.concentration_at_borders = []
            self.derivatives_at_borders_a = []
            self.derivatives_at_borders_b = []
            self.maxima_a = []
            self.maxima_b = []
        if track_scent_marks:
            self.sm_a = []
            self.sm_b = []
        if track_locations:
            self.locs_a = []
            self.locs_b = []

        # perform the simulation for `time_steps` time steps of length dt
        t: float = 0
        for _ in range(time_steps):
            # update
            self.simulation_step(t)

            # store data
            for should_track, update_data in zip(
                    [track_border, track_scent_marks, track_locations], 
                    [self.update_border_data, self.update_scent_marks_data, self.update_location_data]):
                if should_track:
                    update_data()
            
            # increment time
            t += self.sm.dt


    def simulation_step(self, t: float) -> None:
        """Calls the methods of `Wolf` and `ScentMarks` in appropriate order
        to ensure the simulation follows the prescribed mathematical model.

        Args:
            t (float): time
        """
        # update scent marking function
        self.sm.update([self.wolf_a, self.wolf_b])
                
        # compute olfactory interaction values
        self.wolf_a.interaction(self.sm.get_foreign_scent_concentration(self.wolf_a))
        self.wolf_b.interaction(self.sm.get_foreign_scent_concentration(self.wolf_b))
        
        # wolf movement
        self.wolf_a.step(t)
        self.wolf_b.step(t)
        
        # move the dens (no effect in current simulations)
        self.wolf_a.den_movement_process(self.wolf_a)
        self.wolf_b.den_movement_process(self.wolf_b)   

    
    def update_border_data(self) -> None:
        """Stores the border data of current time step -> border location, scent concentration 
        at the border and derivatives of the scent marking function at the border. 
        """
        def calculate_crossing_point(sm_a: np.ndarray, sm_b: np.ndarray) -> Optional[float]:
            """Calculates the border location (as a decimal index - location between two spatial indexes) ,
            that is mean of locations where scent marking functions equal and are non-zero.

            Args:
                sm_a (np.ndarray): scent marking function of `self.wolf_a`
                sm_b (np.ndarray): scent marking function of `self.wolf_b`

            Returns:
                Optional[float]: decimal index of the location (point between two spatial indexes where the border lies).
                    `None` if there is no such place.
            """
            # list of indexes where the scent marking functions equal
            crossing_point: list[float] = []        

            for i in range(len(sm_a) - 1):
                if sm_a[i] < 1e-4 or sm_b[i] < 1e-4:    # at least one of the scent marking functions is zero
                    continue
                
                # scent marking functions do not cross between i and i+1
                if not ((sm_a[i] > sm_b[i] and sm_a[i + 1] < sm_b[i + 1]) or    
                        (sm_b[i] > sm_a[i] and sm_b[i + 1] < sm_a[i + 1])):
                    continue
                
                # scent marking functions do cross between i and i+1, calculate the exact crossing point
                A = np.array([[sm_a[i + 1] - sm_a[i], -1],
                            [sm_b[i + 1] - sm_b[i], -1]])
                b = np.array([i*(sm_a[i + 1] - sm_a[i]) - sm_a[i], i*(sm_b[i + 1] - sm_b[i]) - sm_b[i]])
                crossing_point.append(np.linalg.solve(A, b)[0])
            
            # if there were crossing points, return their mean -> border location
            if crossing_point:
                return np.mean(crossing_point)
            
            return None
        

        def calculate_scent_concentration_at_border(border: Optional[float], sm_a: np.ndarray, sm_b: np.ndarray) -> Optional[float]:
            """Calculates the scent marking function value at the border location (average of the two).

            Args:
                crossing_point (Optional[float]): border location
                sm_a (np.ndarray): scent marking function of `self.wolf_a`
                sm_b (np.ndarray): scent marking function of `self.wolf_b`
            Returns:
                Optional[float]: pheromone function value at the border location.
                    `None` if border location is `None`
            """
            def concentration_at_location(loc: float, sm: np.ndarray) -> float:
                """Calculates scent concentration of passed `sm` function at `loc`

                Args:
                    loc (float): location to evaluate scent concentration at
                    sm (np.ndarray): scent marking function

                Returns:
                    float: scent concentration at `loc`
                """
                i, frac = int(loc), loc % 1
                return sm[i] + frac*(sm[i + 1] - sm[i])
            
            if border is None:
                return None

            # average scent concentration at border
            return (concentration_at_location(border, sm_a) + concentration_at_location(border, sm_b)) / 2
        

        def calculate_derivative_at_location(loc: Optional[float], sm: np.ndarray) -> Optional[float]:
            """Calculates derivative of the scent marking function `sm` at location `loc`

            Args:
                loc (float): location to evaluate derivative at
                sm (np.ndarray): scent marking function

            Returns:
                float: derivative at `loc`
            """
            if loc is None:
                return None
            i: int = int(loc)
            return sm[i+1] - sm[i]
        # get needed fields
        sm_a: np.ndarray = self.sm.get_scent_field(self.wolf_a)
        sm_b: np.ndarray = self.sm.get_scent_field(self.wolf_b)
        border: float = calculate_crossing_point(sm_a, sm_b)
        
        # calculate and store the data
        self.borders.append(border / self.sm.M if border is not None else None)     # convert border location index to actual location
        self.concentration_at_borders.append(calculate_scent_concentration_at_border(border, sm_a, sm_b))
        self.derivatives_at_borders_a.append(calculate_derivative_at_location(border, sm_a))
        self.derivatives_at_borders_b.append(calculate_derivative_at_location(border, sm_b))

    def update_scent_marks_data(self) -> None:  
        "Stores the scent marking functions at current time step"
        self.sm_a.append(self.sm.get_scent_field(self.wolf_a))
        self.sm_b.append(self.sm.get_scent_field(self.wolf_b))
    
    def update_location_data(self) -> None:     
        "Stores the locations of the wolves at current time step"
        self.locs_a.append(self.wolf_a.x)
        self.locs_b.append(self.wolf_b.x)
    
    def get_results(self, time_steps: int, period_length: int, tracked_border: bool = False,
                    tracked_scent_marks: bool = False, tracked_locations: bool = False
                    ) -> dict[str, np.ndarray]:
        """Returns results from the simulations at a form of aggregated values for each time period

        Args:
            time_steps (int): how many time steps did the simulation run for
            period_length (int): length of the period for aggregation (in number time steps)
            tracked_border (bool, optional): whether the border data was stored during the simulation. 
                Defaults to False.
            tracked_scent_marks (bool, optional): whether the scent marking function data
                was stored during the simulation. 
                Defaults to False.
            tracked_locations (bool, optional): whether the locations of the wolves were stored
                during the simulation.
                Defaults to False.

        Returns:
            dict[str, np.ndarray]: aggregated result, for each statistic one value for each period
        """
        res: dict[str, np.ndarray] = {}

        # apply aggregating func to each section of time steps corresponding to one period
        func_through_periods: Callable[[np.ndarray, np.ndarray]] = lambda arr, period_times, func: (
            [func(np.array(arr[intervals[i - 1]: intervals[i]], dtype=np.float16), 
                  axis=0)
             for i in range(1, len(period_times))]
        )

        # define intervals for aggregation
        intervals: np.ndarray = np.arange(0, time_steps + 1, period_length)   
        res["tick"] = intervals[1:]
        
        # aggregate border values
        if tracked_border:
            res["border"] = func_through_periods(self.borders, intervals, np.nanmean)
            res["border_sm"] = func_through_periods(self.concentration_at_borders, intervals, np.nanmean)
            res["der_a"] = func_through_periods(self.derivatives_at_borders_a, intervals, np.nanmean)
            res["der_b"] = func_through_periods(self.derivatives_at_borders_b, intervals, np.nanmean)

        # aggregate scent marking function values
        if tracked_scent_marks:
            res["sm_a"] = func_through_periods(self.sm_a, intervals, np.nanmean)
            res["sm_b"] = func_through_periods(self.sm_b, intervals, np.nanmean)
        
        # aggregate location values
        if tracked_locations:
            # maximum deviation from den
            res["max_x_a"] = func_through_periods(self.locs_a, intervals, np.max) 
            res["min_x_b"] = func_through_periods(self.locs_b, intervals, np.min) 
            # (buffer zone) boundary beyond which only 5% of time was spent
            res["r_95_a"] = func_through_periods(self.locs_a, intervals, partial(np.quantile, q=0.95))
            res["r_95_b"] = func_through_periods(self.locs_b, intervals, partial(np.quantile, q=0.05))

        return res
    

def perform_simulations(args: tuple[list[config], int]) -> None:
    """Performs simulations for passed configurations

    Args:
        args (tuple[list[config], int]): should be a pair of the next two arguments
            - `configs (list[config])`: list of (L, T, d) configurations 
                to run simulations for
            - `id_start (int)`: id to give the first simulation from the configurations, 
                rest are incremented
    """
    configs, id_start = args

    # database to store the simulated data to
    db: DatabaseHandler = DatabaseHandler("data/simulations.sqlite")
    
    # id to give the starting simulations, how many times to run one configuration 
    # and how many time steps in one simulation
    id: int = id_start
    repetitions: int = 100
    time_steps: int = 50_000
   
    # order of columns in `simulations` table in `db` to write simulated data to
    simulations_table_col_order: list[str] = ["tick", "border", "border_sm", "der_a", "der_b", "max_x_a", "min_x_b", "r_95_a", "r_95_b"]
    
    # run for each config
    for L, T, d in configs:       
        print(f"Running, id={id}, config={(L, T, d)}")

        den_a: float = 0.5 - round((L / 2), 3)
        den_b: float = 0.5 + round((L / 2), 3)
        
        # store data about the configurations and their assigned id
        db.store_runs(np.array([[id, repetitions, time_steps, T, T, den_a, den_b, d]]))
        
        # run simulation with current configuration `repetitions` times
        for it in range(repetitions):
            try:
                # simulation
                s: Simulation = Simulation(*build_instances(
                    den_a=den_a, den_b=den_b, T_a=T, T_b=T, diffusion_coefficient=d
                    ))
                s.simulate(time_steps, track_border=True, track_scent_marks=False, track_locations=True)

                # get results
                res: dict[str, Any] = s.get_results(
                    time_steps=time_steps, period_length=round(T / s.sm.dt), 
                    tracked_border=True, tracked_scent_marks=False, tracked_locations=True
                    )

                matrix: np.ndarray = np.array(
                    [[id for _ in range(len(res["tick"]))], 
                     [it for _ in range(len(res["tick"]))]] + 
                    [res[stat] for stat in simulations_table_col_order])

                # store the results in db
                db.store_simulations(matrix.T)

            except Exception as e:
                print(f"Exception at id={id}, it={it}, config={(L, T, d)}")
                print(e)
        
        # increment the id for the next configuration
        id += 1


def thesis_simulations(processes: int) -> None:
    """Runs simulations for all configurations stated in the thesis

    Args:
        processes (int): how many Python processes to start. 
    """

    def split_array(arr: list[Any], n: int) -> list[list[Any]]:
        """Splits passed array to `n` sublists of almost equal length

        Args:
            arr (list[Any]): array to split
            n (int): desired number of sublists

        Returns:
            list[list[Any]]: `n` sublists of almost equal length
        """
        k, m = divmod(len(arr), n)
        return [arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    
    if processes == 0:
        return
    
    # define all configurations
    L_all: list[float] = [0.2, 0.28, 0.35, 0.43, 0.5, 0.58, 0.65, 0.73, 0.8]
    T_all: list[float] = [0.125, 0.187, 0.25, 0.312, 0.375, 0.437, 0.5]
    d_all: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    all_configs: list[config] = list(product(L_all, T_all, d_all))

    # split them as evenly as possible for each process
    configs_splitted: list[list[config]] = split_array(all_configs, processes)
    start_ids: list[float] = [
        sum(map(len, configs_splitted[:i])) for i in range(len(configs_splitted))
        ]
    
    # run each subset of configurations on a separate process
    with Pool(processes=processes) as pool:
        pool.map(perform_simulations, zip(configs_splitted, start_ids))
    

def main() -> None:
    # create simulation
    s: Simulation = Simulation(*build_instances())
    
    # simulate
    s.simulate(50_000, track_border=True)
    
    # extract gathered data
    res: dict[str, np.ndarray] = s.get_results(
        time_steps=50000, period_length=round(s.wolf_a.T / s.sm.dt),
        tracked_border=True
        )

    # plot border location in time
    plt.plot(res["tick"], res["border"])
    plt.show()
    

if __name__ == "__main__":
    main()
