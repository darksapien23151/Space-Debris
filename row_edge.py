import os
import pandas as pd
import numpy as np
import networkx as nx
from sgp4.api import Satrec, jday
from tqdm import tqdm
import pickle

# Convert DataFrame row to fake but valid TLE
def tle_from_row(row):
    sat_num = int(row["Satellite Number"])
    epoch = pd.to_datetime(row["EPOCH"])
    epoch_year = epoch.year % 100
    epoch_day = (epoch - pd.Timestamp(f"{epoch.year}-01-01")).total_seconds() / 86400 + 1

    mean_motion = float(row["Mean Motion"])
    ecc = float(row["Eccentricity"])
    incl = float(row["Inclination"])
    raan = float(row["RAAN"])
    argp = float(row["Argument of Perigee"])
    mean_anom = float(row["Mean Anomaly"])

    line1 = f"1 {sat_num:05d}U {epoch_year:02d}001A   {epoch_year:02d}{epoch_day:.8f}  .00000000  00000-0  00000-0 0  9993"
    line2 = f"2 {sat_num:05d} {incl:8.4f} {raan:8.4f} {ecc:.7f} {argp:8.4f} {mean_anom:8.4f} {mean_motion:11.8f}00003"
    return line1, line2

# Predict satellite position using SGP4
def get_position(satrec, timestamp):
    jd, fr = jday(timestamp.year, timestamp.month, timestamp.day,
                  timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond / 1e6)
    e, r, v = satrec.sgp4(jd, fr)
    return np.array(r) if e == 0 else None

# Load all satellite CSVs from a folder
def load_all_satellites(folder):
    sats = {}
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            try:
                sat_id = int(file.split('_')[-1].split('.')[0])
                df = pd.read_csv(os.path.join(folder, file))
                df['EPOCH'] = pd.to_datetime(df['EPOCH'])
                sats[sat_id] = df
            except:
                continue
    return sats

# Build a graph for a given timestamp
def build_graph_at_timestamp(sat_dict, timestamp, threshold_km=50000): 
    G = nx.Graph()
    positions = {}

    for sat_id, df in sat_dict.items():
        row = df.iloc[(df['EPOCH'] - timestamp).abs().argsort()[:1]]
        if row.empty: continue
        row = row.iloc[0]

        try:
            line1, line2 = tle_from_row(row)
            satrec = Satrec.twoline2rv(line1, line2)
            pos = get_position(satrec, timestamp)
            if pos is not None:
                G.add_node(sat_id, position=pos)
                positions[sat_id] = pos
        except:
            continue

    ids = list(positions.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = np.linalg.norm(positions[ids[i]] - positions[ids[j]])
            if d < threshold_km:
                G.add_edge(ids[i], ids[j])
    return G

# Build the entire DTDG
def build_dtdg(sat_dict, start_time, end_time, freq='6H'):
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    graphs = []
    for t in tqdm(timestamps, desc="Building DTDG"):
        g = build_graph_at_timestamp(sat_dict, t)
        graphs.append((t, g))
    return graphs

# Save graphs as pickle
def save_graphs(graphs, output_path='dtdg_graphs.pkl'):
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f)

# ========== MAIN AUTOMATION ==========
if __name__ == "__main__":
    folder_path = "parsed_clean_edges"  
    sat_data = load_all_satellites(folder_path)

    # Choose sensible global range
    start_time = pd.Timestamp("2009-10-19 00:00:00")
    end_time   = pd.Timestamp("2023-12-31 00:00:00")

    dtdg_graphs = build_dtdg(sat_data, start_time, end_time, freq="6H")

    # 💾 Save them
    save_graphs(dtdg_graphs, "output_dtdg_2graphs.pkl")

    print("✅ DTDG generated and saved as 'output_dtdg_graphs.pkl'")
