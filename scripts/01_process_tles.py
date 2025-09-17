import pandas as pd
from tqdm import tqdm
import sys

def calculate_checksum(tle_line: str) -> int:
    """Calculates the modulo-10 checksum for a given TLE line."""
    checksum_sum = 0
    for char in tle_line[:68]:
        if char.isdigit():
            checksum_sum += int(char)
        elif char == '-':
            checksum_sum += 1
    return checksum_sum % 10

def parse_tle_for_sgp4(file_path: str) -> pd.DataFrame:
    """
    Parses a TLE file, validating entries and saving the raw lines needed by sgp4.
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
        return pd.DataFrame()

    parsed_data = []
    
    for i in tqdm(range(0, len(lines), 3), desc="Parsing TLE File"):
        if i + 2 < len(lines):
            name, line1, line2 = lines[i], lines[i+1], lines[i+2]
            try:
                # Validate checksums before proceeding
                if (calculate_checksum(line1) != int(line1[68]) or
                    calculate_checksum(line2) != int(line2[68])):
                    continue
            except (ValueError, IndexError):
                continue
            try:
                # Extract only the data needed for the next phase
                data_entry = {
                    'name': name,
                    'norad_id': int(line1[2:7]),
                    'mean_motion': float(line2[52:63]),
                    'line1': line1,
                    'line2': line2
                }
                parsed_data.append(data_entry)
            except (ValueError, IndexError):
                continue
    
    if not parsed_data:
        print("Error: No valid TLE entries were parsed.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    tle_file = 'active.txt'
    clean_data_file = 'sats_df_clean_sgp4.csv' # New name to avoid confusion

    print("--- Starting Phase 1: Robust TLE Parsing for SGP4 ---")
    sats_df = parse_tle_for_sgp4(tle_file)
    
    if not sats_df.empty:
        sats_df.to_csv(clean_data_file, index=False)
        print(f"\n✅ Phase 1 Complete: Parsed {len(sats_df)} satellites and saved to '{clean_data_file}'")
    else:
        print("\n❌ Phase 1 Failed: No valid data was parsed.")
