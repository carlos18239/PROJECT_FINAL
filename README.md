# Federated Learning - Experiment Execution

This project implements federated learning experiments in two modes: semi-decentralized and centralized. Below are the steps to run the agents and server in each mode, using the provided IPs and commands.

## Prerequisites

- Have `conda` installed and the `federatedenv2` environment set up.
- Clone this repository and navigate to the appropriate folder.
- Install all required dependencies in the environment.

## Semi-Decentralized Execution

### Activate Environment

```bash
conda activate federatedenv2
```

### Run Agents (use IP 172.23.207.49)

In the `semie_edescentralized/` folder, run the following commands in separate terminals:

```bash
python3 agent_csv.py agent_1 data1.csv ws://172.23.207.49:8765
python3 agent_csv.py agent_2 data2.csv ws://172.23.207.49:8765
python3 agent_csv.py agent_3 data3.csv ws://172.23.207.49:8765
python3 agent_csv.py agent_4 data3.csv ws://172.23.207.49:8765
```

## Centralized Execution

### Server (use IP 172.23.207.55)

In the `centralized/` folder, run:

```bash
python server_centralized.py
```

### Agents (Raspberry Pi)

In the `centralized/` folder, run the following on each Raspberry Pi:

- **Raspberry Pi 1:**
  ```bash
  python worker_centralized.py agent_1 data1.csv ws://172.23.207.55:8765
  ```
- **Raspberry Pi 2:**
  ```bash
  python worker_centralized.py agent_2 data2.csv ws://172.23.207.55:8765
  ```
- **Raspberry Pi 3:**
  ```bash
  python worker_centralized.py agent_3 data3.csv ws://172.23.207.55:8765
  ```
- **Raspberry Pi 4:**
  ```bash
  python worker_centralized.py agent_4 data4.csv ws://172.23.207.55:8765
  ```

## Notes

- Make sure the IPs are accessible from the devices running the agents and server.
- The CSV files (`data1.csv`, `data2.csv`, etc.) must be present in the corresponding folder.
- The environment must be activated before running the scripts.

## Folder Structure

- `semie_edescentralized/`: Scripts and data for the semi-decentralized mode.
- `centralized/`: Scripts and data for the centralized mode.

## Contact

For questions or support, contact the repository maintainer.
