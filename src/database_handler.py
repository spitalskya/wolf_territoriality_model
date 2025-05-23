import sqlite3
import numpy as np
import pandas as pd


class DatabaseHandler:
    """Class that handles to communication with database of simulated data

    Attributes:
        db_path (str): path to a SQLITE database file
    """
    
    db_path: str
    
    def __init__(self, db_path="data/simulations.sqlite") -> None:
        """Creates the tables necessary for storing data from simulations

        Args:
            db_path (str, optional): path to a SQLITE database file. 
                Defaults to "data/simulations.sqlite".
        """
        self.db_path = db_path

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # table to store what configurations were run and their ids
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER,
                    k INTEGER,                      -- number of repetitions of one configuration
                    n INTEGER,                      -- number of time steps in one simulation
                    period_a REAL,
                    period_b REAL,
                    den_a REAL,
                    den_b REAL,
                    diffusion_coefficient REAL
                )
            """)

            # table to store simulated data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER,             -- id of the run (defines configuration)
                    it INTEGER,             -- index of this simulation among the k repetitions
                    tick INTEGER,           -- time representing the end of the aggregation interval
                    border REAL,            -- mean location of the border during interval [tick - T, tick]
                    border_sm REAL,         -- mean concentration of scent at the border during interval
                    der_a REAL,             -- mean derivative of the scent marking function of wolf A at the border during interval
                    der_b REAL,             -- mean derivative of the scent marking function of wolf B at the border during interval
                    max_x_a REAL,           -- location of the maximum deviation (to the right) from the den of the wolf A during interval
                    min_x_b REAL,           -- location of the maximum deviation (to the left) from the den of the wolf B during interval
                    r_95_a REAL,            -- 95% quantile of the locations of the wolf A during interval
                    r_95_b REAL             -- 5% quantile of the locations of the wolf A during interval
                )
            """)

            # table storing aggregated statistics from the final 10% of simulation steps for each configuration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER,             -- id of the run (defines configuration)
                    L REAL,                 -- distance of the dens
                    T REAL,                 -- harmonic mean of the periods of the movement weighing function of the wolves
                    d REAL,                 -- diffusion coefficient
                    border_mean REAL,       -- mean location of the border
                    border_msd REAL,        -- mean square displacement of the border location
                    buffer_width REAL       -- mean buffer zone width
                )
            """)
            conn.commit()
    
    def store_runs(self, matrix: np.ndarray) -> None:
        """Stores passed matrix of runs into the table `runs`

        Args:
            matrix (np.ndarray): matrix of data for `runs` table
        """        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()            
            rows = [tuple(row) for row in matrix]
            cursor.executemany(
                """
                INSERT INTO runs (
                    id, k, n, period_a, period_b, 
                    den_a, den_b, diffusion_coefficient) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
            conn.commit()
    
    
    def store_simulations(self, matrix: np.ndarray) -> None:
        """Stores passed matrix of simulations data into the table `simulations`

        Args:
            matrix (np.ndarray): matrix of data for `simulations` table
        """   
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()            
            rows = [tuple(row) for row in matrix]
            cursor.executemany(
                """
                INSERT INTO simulations (
                    id,it,tick,
                    border, border_sm, der_a, der_b,
                    max_x_a, min_x_b, r_95_a, r_95_b) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
            conn.commit()
            
    
    def fill_statistics(self) -> None:
        "Fills the `statistics` table (only for ids that do now have already statistics computed)"        
        with sqlite3.connect(self.db_path) as conn:
            simulations: pd.DataFrame = pd.read_sql_query(
                """
                SELECT * 
                FROM simulations sim
                JOIN runs r USING (id)
                WHERE NOT EXISTS (
                    SELECT 1 FROM statistics st
                    WHERE st.id = sim.id
                )
                """,
                con=conn)
            
            # keep only last 10% of each simulation
            simulations = simulations[simulations["tick"] >= max(simulations["tick"])*0.9]

            # perform needed transformations
            simulations["T"] = (2*simulations["period_a"]*simulations["period_b"]) / (simulations["period_a"] + simulations["period_b"])
            simulations["L"] = simulations["den_b"] - simulations["den_a"]
            simulations["bw"] = (simulations["r_95_b"] - simulations["r_95_a"]).clip(lower=0)

            # keep only the columns needed
            simulations = simulations[["id", "border", "bw", "T", "L", "diffusion_coefficient"]]           

            # group by id, compute means and variance of the border location (mean square displacement)                        
            simulations_grouped: pd.DataFrame = simulations.groupby("id", as_index=False).mean()
            simulations_grouped["border_msd"] = simulations[["id", "border"]].groupby("id", as_index=False).var(ddof=0)["border"]
            
            # store the data into `statistics` table
            rows = [tuple(row) for _, row in simulations_grouped[["id", "L", "T", "diffusion_coefficient", "border", "border_msd", "bw"]].iterrows()]
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO statistics (id, L, T, d, border_mean, border_msd, buffer_width)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows)
            conn.commit()
            

def main() -> None:
    # fill the `statistics` table
    dbh: DatabaseHandler = DatabaseHandler()
    dbh.fill_statistics()
    
if __name__ == "__main__":
    main()
