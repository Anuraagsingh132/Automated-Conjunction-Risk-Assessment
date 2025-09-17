import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import requests
import joblib
import os
import multiprocessing
from functools import partial

# Import SGP4 for propagation
from sgp4.api import Satrec
from astropy.time import Time

# --- CONFIGURATION ---
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
LOCAL_TLE_FILENAME = 'active.txt' 
MODEL_FILENAME = 'conjunction_model.joblib'
POTENTIAL_EVENTS_FILENAME = 'potential_conjunctions.csv' # File to save coarse search results
SIM_DURATION_DAYS = 3.0
NUM_SATELLITES_TO_PROCESS = 2000
COARSE_TIME_STEP_MIN = 1.0
DETECTION_THRESHOLD_KM = 25.0

# --- DATA ACQUISITION & PARSING (SERIAL) ---
def get_tle_data(url: str, filename: str) -> bool:
    if os.path.exists(filename):
        print(f"--- Found local TLE file: '{filename}' ---")
        return True
    print(f"--- Local file not found. Downloading latest TLE data from CelesTrak ---")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if not response.text.strip().startswith('1 '):
            print("Error: Downloaded file is not a valid TLE file.", file=sys.stderr)
            return False
        with open(filename, 'w') as f: f.write(response.text)
        print(f"Successfully downloaded and saved to '{filename}'")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download TLE data. {e}", file=sys.stderr)
        return False

def calculate_checksum(tle_line: str) -> int:
    s = 0
    for char in tle_line[:68]:
        if char.isdigit(): s += int(char)
        elif char == '-': s += 1
    return s % 10

def parse_tle_for_sgp4(file_path: str) -> pd.DataFrame:
    print("--- Parsing TLE file ---")
    try:
        with open(file_path, 'r') as f: lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError: return pd.DataFrame()
    parsed_data = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name, line1, line2 = lines[i], lines[i+1], lines[i+2]
            try:
                if (calculate_checksum(line1) != int(line1[68]) or calculate_checksum(line2) != int(line2[68])): continue
                parsed_data.append({'name': name, 'norad_id': int(line1[2:7]), 'mean_motion': float(line2[52:63]), 'line1': line1, 'line2': line2})
            except (ValueError, IndexError): continue
    return pd.DataFrame(parsed_data)

# --- PARALLEL WORKER FUNCTIONS ---

def coarse_search_worker(pair_indices: list, leo_df: pd.DataFrame, jds: np.ndarray, all_positions: np.ndarray, threshold_km: float) -> list:
    """Worker function to check a chunk of satellite pairs for potential conjunctions."""
    potential_events = []
    for i, j in pair_indices:
        delta_pos = all_positions[i] - all_positions[j]
        distances = np.linalg.norm(delta_pos, axis=1)
        if np.min(distances) < threshold_km:
            min_dist_idx = np.argmin(distances)
            potential_events.append({
                'id_A': leo_df.iloc[i]['norad_id'], 
                'id_B': leo_df.iloc[j]['norad_id'],
                'approx_tca_jd': jds[min_dist_idx]
            })
    return potential_events

def refine_and_predict_worker(event: dict, leo_df_dict: dict, model, feature_names: list) -> dict or None:
    """Worker function to refine one event and predict its risk."""
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
    
    features = {
        'tca_jd': jds_fine[min_idx], 'miss_distance_km': min_dist_km,
        'id_A': event['id_A'], 'id_B': event['id_B'],
        'relative_velocity_km_s': np.linalg.norm(v_A_tca - v_B_tca),
        'inclination_diff': abs(np.rad2deg(sat_A.inclo) - np.rad2deg(sat_B.inclo)),
        'eccentricity_diff': abs(sat_A.ecco - sat_B.ecco),
        'raan_diff': abs(np.rad2deg(sat_A.nodeo) - np.rad2deg(sat_B.nodeo)) % 180
    }

    features_df = pd.DataFrame([features])
    X_predict = features_df[feature_names]
    prediction = model.predict(X_predict)[0]

    if prediction == 1:
        features['probability'] = model.predict_proba(X_predict)[0][1]
        return features
    return None

# --- MAIN INFERENCE WORKFLOW ---
def run_inference_pipeline():
    if not get_tle_data(TLE_URL, LOCAL_TLE_FILENAME): return
    sats_df = parse_tle_for_sgp4(LOCAL_TLE_FILENAME)
    if sats_df.empty:
        print("\n❌ Process Halted: No valid TLE data parsed.", file=sys.stderr); return

    print(f"--- Loading trained model from '{MODEL_FILENAME}' ---")
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found.", file=sys.stderr); return

    leo_df = sats_df[sats_df['mean_motion'] > 11.25].head(NUM_SATELLITES_TO_PROCESS).reset_index(drop=True)
    
    sats_list = [Satrec.twoline2rv(row['line1'], row['line2']) for _, row in leo_df.iterrows()]
    if not sats_list:
        print("\n❌ Process Halted: Could not create satellite objects.", file=sys.stderr); return

    print(f"--- Propagating {len(sats_list)} LEO satellites for {SIM_DURATION_DAYS} days ---")
    start_jd = sats_list[0].jdsatepoch
    jds = np.arange(start_jd, start_jd + SIM_DURATION_DAYS, COARSE_TIME_STEP_MIN / 1440.0)
    all_positions = np.array([sat.sgp4_array(np.floor(jds), jds - np.floor(jds))[1] for sat in tqdm(sats_list, desc="Propagating Orbits")])

    print("--- Performing parallel coarse search for potential conjunctions ---")
    num_sats = len(sats_list)
    pair_indices = [(i, j) for i in range(num_sats) for j in range(i + 1, num_sats)]
    
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} worker processes.")
    
    chunk_size = len(pair_indices) // num_processes
    chunks = [pair_indices[i:i + chunk_size] for i in range(0, len(pair_indices), chunk_size)]
    
    worker_func = partial(coarse_search_worker, leo_df=leo_df, jds=jds, all_positions=all_positions, threshold_km=DETECTION_THRESHOLD_KM)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, chunks), total=len(chunks), desc="Coarse Search"))
    
    potential_conjunctions = [event for sublist in results for event in sublist]

    # --- THIS IS THE ADDED SECTION ---
    if potential_conjunctions:
        print(f"\n--- Saving {len(potential_conjunctions)} potential events to '{POTENTIAL_EVENTS_FILENAME}' ---")
        pd.DataFrame(potential_conjunctions).to_csv(POTENTIAL_EVENTS_FILENAME, index=False)
    # --- END OF ADDED SECTION ---

    print(f"\n--- Found {len(potential_conjunctions)} potential events. Refining and predicting risk in parallel... ---")
    leo_df_dict = leo_df.set_index('norad_id').to_dict('index')
    try:
        feature_names = model.get_booster().feature_names
    except AttributeError:
        feature_names = ['miss_distance_km', 'relative_velocity_km_s', 'inclination_diff', 'eccentricity_diff', 'raan_diff']
    
    predict_worker_func = partial(refine_and_predict_worker, leo_df_dict=leo_df_dict, model=model, feature_names=feature_names)
    
    high_risk_alerts = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(predict_worker_func, potential_conjunctions), total=len(potential_conjunctions), desc="Predicting Risk"):
            if result:
                high_risk_alerts.append(result)

    if high_risk_alerts:
        high_risk_alerts.sort(key=lambda x: x['tca_jd'])
        print(f"\n--- GENERATED {len(high_risk_alerts)} HIGH-RISK ALERTS (sorted by time) ---")
        for alert in high_risk_alerts:
            print("\n" + "="*50 + f"\n🚨 HIGH-RISK CONJUNCTION ALERT! 🚨\n" + "="*50)
            tca_time = Time(alert['tca_jd'], format='jd').iso
            print(f"  - Satellite A (NORAD ID): {alert['id_A']}")
            print(f"  - Satellite B (NORAD ID): {alert['id_B']}")
            print(f"  - Time of Closest Approach (TCA): {tca_time} UTC")
            print(f"  - Predicted Miss Distance: {alert['miss_distance_km']:.2f} km")
            print(f"  - AI-Predicted Risk Probability: {alert['probability']*100:.1f}%")
            print("="*50 + "\n")
            
    print(f"\n✅ Inference Pipeline Complete. Found {len(high_risk_alerts)} high-risk events.")

if __name__ == "__main__":
    run_inference_pipeline()
