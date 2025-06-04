## Running the TORCS AI Racing Controller

### Prerequisites
- Python 3.x installed
- Required Python packages (install using `pip install -r requirements.txt`)
- TORCS game installed

### Step-by-Step Running Instructions

1. **Start TORCS Game**
   - Navigate to your TORCS installation directory
   - Run `wtorcs.exe`
   - In the game menu:
     - Select "Race" → "Quick Race"
     - Choose a track
     - Make sure "SCR" is selected as one of the drivers
     - Click "Start Race"
   - Wait for the game to load

2. **Run the AI Controller**
   - Open a new terminal/command prompt
   - Navigate to the Aiproject-main directory:
     ```bash
     cd path/to/Aiproject-main
     ```
   - Run the model client:
     ```bash
     python model_client.py

3. **Controls During Race**
   - Press 'M' to toggle between manual and AI control
   - Press 'S' to start logging data
   - Press 'E' to stop logging data
   - Press 'Q' to quit

4. **Manual Control Keys**
   - Arrow Keys:
     - Up: Accelerate
     - Down: Brake/Reverse
     - Left/Right: Steer
   - Z: Shift up
   - X: Shift down

### Troubleshooting
- If the connection fails, ensure TORCS is running and the race has started
- Check that the SCR driver is selected in TORCS
- Verify all required Python packages are installed

### Data Collection
- Data is temporarily stored in `temp_run.csv`
- After the session, you'll be prompted to save the data to `model_test_data.csv`
- Type 'y' to save or 'n' to discard the data
