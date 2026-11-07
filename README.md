# FloodImpactNet — local application

The project has a React/Vite frontend and a Flask inference backend. Keep the app inside `D:\FloodImpactNet` beside the three unzipped UI-asset folders so the backend can discover the Module 1 checkpoint automatically.

## First-time setup

Open PowerShell window 1:

```powershell
cd D:\FloodImpactNet\FloodImpactNet_App\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open PowerShell window 2:

```powershell
cd D:\FloodImpactNet\FloodImpactNet_App\frontend
npm install
npm run dev
```

Open `http://localhost:5173` in a browser. Leave both PowerShell windows running.

After the first setup, activate the backend environment and run `python app.py`, then run `npm run dev` in the frontend.

## What is live now

- Module 1 accepts an 8-band, 512 × 512 GeoTIFF and runs local U-Net inference.
- Modules 2 and 3 present the validated scenario and uncertainty outputs packaged from the notebook.
- The interface labels synthetic scenarios and relative uncertainty clearly; it does not claim observed damage or calibrated risk.

Deployment is intentionally deferred until the local deliverable is accepted.
