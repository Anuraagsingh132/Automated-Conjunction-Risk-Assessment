import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import multiprocessing
from functools import partial

# Import SGP4 and Astropy for calculations
from sgp4.api import Satrec
from astropy.time import Time

# --- SIMULATION PARAMETERS ---
NUM_SATELLITES_TO_SIMULATE = 6000
SIM_DURATION_DAYS = 10.0
COARSE_TIME_STEP_MIN = 1.0 
DETECTION_THRESHOLD_KM = 25.0

# --- DATA PREPARATION (SERIAL) ---
def parse_tle_for_sgp4(file_path: str) -> pd.DataFrame:
    """Parses a TLE file, validating entries and saving the raw lines needed by sgp4."""
    print("--- Parsing TLE file ---")
    try:
        with open(file_path, 'r') as f: lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
        return pd.DataFrame()

    parsed_data = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name, line1, line2 = lines[i], lines[i+1], lines[i+2]
            try:
                if (line1[0] != '1' or line2[0] != '2'): continue
                parsed_data.append({'name': name, 'norad_id': int(line1[2:7]), 'mean_motion': float(line2[52:63]), 'line1': line1, 'line2': line2})
            except (ValueError, IndexError): continue
    return pd.DataFrame(parsed_data)

# --- PARALLEL WORKER FUNCTIONS ---

def coarse_search_worker(pair_indices: list, leo_df: pd.DataFrame, jds: np.ndarray, threshold_km: float) -> list:
    """
    Worker function that propagates only the necessary orbits and then checks
    its assigned chunk of satellite pairs for potential conjunctions.
    """
    potential_events = []
    
    # Find the unique satellite indices this worker needs to propagate
    needed_indices = sorted(list(set([i for i, j in pair_indices] + [j for i, j in pair_indices])))
    
    # Create and propagate only the required satellites
    local_sats = {idx: Satrec.twoline2rv(leo_df.iloc[idx]['line1'], leo_df.iloc[idx]['line2']) for idx in needed_indices}
    
    jd_whole = np.floor(jds)
    jd_frac = jds - jd_whole
    local_positions = {idx: sat.sgp4_array(jd_whole, jd_frac)[1] for idx, sat in local_sats.items()}

    # Perform the coarse search on the locally propagated data
    for i, j in pair_indices:
        pos_A = local_positions[i]
        pos_B = local_positions[j]
        
        distances = np.linalg.norm(pos_A - pos_B, axis=1)
        if np.min(distances) < threshold_km:
            min_dist_idx = np.argmin(distances)
            potential_events.append({
                'id_A': leo_df.iloc[i]['norad_id'], 
                'id_B': leo_df.iloc[j]['norad_id'],
                'approx_tca_jd': jds[min_dist_idx]
            })
    return potential_events

def refine_and_engineer_worker(event: dict, leo_df_dict: dict) -> dict or None:
    """Worker function to refine one event and engineer its features."""
    sat_A_data = leo_df_dict[event['id_A']]
    sat_B_data = leo_df_dict[event['id_B']]
    sat_A = Satrec.twoline2rv(sat_A_data['line1'], sat_A_data['line2'])
    sat_B = Satrec.twoline2rv(sat_B_data['line1'], sat_B_data['line2'])
    
    jd_approx = event['approx_tca_jd']
    jds_fine = np.arange(jd_approx - (1.5 / 1440.0), jd_approx + (1.5 / 1440.0), 1.0 / 86400.0)
    
    _, r_A, v_A = sat_A.sgp4_array(np.floor(jds_fine), jds_fine - np.floor(jds_fine))
    _, r_B, v_B = sat_B.sgp4_array(np.floor(jds_fine), jds_fine - np.floor(jds_fine))
    
    distances_fine = np.linalg.norm(r_A - r_B, axis=1)
    min_dist_km = np.min(distances_fine)

    if min_dist_km >= 10.0: return None
    
    min_idx = np.argmin(distances_fine)
    v_A_tca, v_B_tca = v_A[min_idx], v_B[min_idx]
    
    return {
        'tca_jd': jds_fine[min_idx], 'miss_distance_km': min_dist_km,
        'id_A': event['id_A'], 'id_B': event['id_B'],
        'relative_velocity_km_s': np.linalg.norm(v_A_tca - v_B_tca),
        'inclination_diff': abs(np.rad2deg(sat_A.inclo) - np.rad2deg(sat_B.inclo)),
        'eccentricity_diff': abs(sat_A.ecco - sat_B.ecco),
        'raan_diff': abs(np.rad2deg(sat_A.nodeo) - np.rad2deg(sat_B.nodeo)) % 180,
        'is_high_risk': 1 if min_dist_km < 1.0 else 0
    }

# --- MAIN WORKFLOW ---
def main():
    clean_data_file = 'sats_df_clean_sgp4.csv'
    final_dataset_file = 'conjunction_events_sgp4.csv'

    print("--- Starting Phase 2: Conjunction Analysis with SGP4 (Parallel) ---")
    sats_df = parse_tle_for_sgp4('active.txt') # Assumes Phase 1 has run or active.txt exists
    if sats_df.empty: return

    leo_df = sats_df[sats_df['mean_motion'] > 11.25].head(NUM_SATELLITES_TO_SIMULATE).reset_index(drop=True)
    print(f"Preparing to simulate {len(leo_df)} LEO satellites...")

    # Determine a common time array for all workers
    temp_sat = Satrec.twoline2rv(leo_df.iloc[0]['line1'], leo_df.iloc[0]['line2'])
    start_jd = temp_sat.jdsatepoch
    jds = np.arange(start_jd, start_jd + SIM_DURATION_DAYS, COARSE_TIME_STEP_MIN / 1440.0)

    print("--- Performing parallel coarse search for potential conjunctions ---")
    num_sats = len(leo_df)
    pair_indices = [(i, j) for i in range(num_sats) for j in range(i + 1, num_sats)]
    
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} worker processes.")
    
    chunk_size = len(pair_indices) // num_processes
    chunks = [pair_indices[i:i + chunk_size] for i in range(0, len(pair_indices), chunk_size)]
    
    worker_func = partial(coarse_search_worker, leo_df=leo_df, jds=jds, threshold_km=DETECTION_THRESHOLD_KM)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, chunks), total=len(chunks), desc="Coarse Search"))
    
    potential_conjunctions = [event for sublist in results for event in sublist]

    print(f"\n--- Found {len(potential_conjunctions)} potential events. Refining in parallel... ---")
    leo_df_dict = leo_df.set_index('norad_id').to_dict('index')
    
    refine_worker_func = partial(refine_and_engineer_worker, leo_df_dict=leo_df_dict)
    
    final_dataset = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(refine_worker_func, potential_conjunctions), total=len(potential_conjunctions), desc="Refining Events"):
            if result:
                final_dataset.append(result)

    if final_dataset:
        pd.DataFrame(final_dataset).to_csv(final_dataset_file, index=False)
        print(f"\n✅ Phase 2 Complete! Generated dataset with {len(final_dataset)} labeled events.")
        print(f"Final dataset saved to '{final_dataset_file}'")
    else:
        print("\n❌ Process Finished, but no conjunction events met the final criteria.")

if __name__ == "__main__":
    main()
