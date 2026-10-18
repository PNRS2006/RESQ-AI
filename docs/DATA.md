# ResQ-AI Data Specification & Synthetic Methodology

## 1. Transparency & Synthetic Data Declaration

> **IMPORTANT ETHICAL DISCLOSURE:**  
> All incident records, coordinates, casualty estimates, and risk labels used in ResQ-AI are **synthetic / simulated**.  
> In compliance with privacy best practices and hackathon ethical standards, **no personally identifiable information (PII)** or sensitive operational dispatch logs from real emergency services are stored or utilized.  
> Every empirical number, latency measurement, and evaluation score presented in this project was produced by executing real code against this simulated benchmark.

---

## 2. Why Simulated Data?

Emergency services (police, fire, EMS) rightly treat CAD dispatch logs and caller records as protected data containing private medical histories, victim identities, and precise residential addresses. Furthermore, rare catastrophic events—such as major structural collapses or massive flood rescues—occur infrequently in historical public datasets, creating severe class imbalance.

Our simulator models realistic crisis dynamics while respecting strict data privacy.

---

## 3. Mathematical Formulation of Ground-Truth Synthetic Risk

The ground-truth continuous risk score $R \in [0.0, 1.0]$ is computed using a multi-factor physical formula modulated by Gaussian noise:

$$R = R_{\text{severity}} + R_{\text{casualties}} + R_{\text{type}} + R_{\text{weather}} + R_{\text{traffic}} + R_{\text{road}} + R_{\text{time}} + \epsilon$$

Where $\epsilon \sim \mathcal{N}(0, 0.04)$ and the individual components are defined as:

1. **Severity Component (35% max weight):**
   $$R_{\text{severity}} = \left(\frac{\text{severity}}{5.0}\right) \times 0.35$$

2. **Casualty Component (25% max weight):**
   $$R_{\text{casualties}} = \min\left(\frac{\text{people\_affected}}{30.0}, 1.0\right) \times 0.25$$

3. **Intrinsic Danger by Incident Type (15% max weight):**
   - Building Collapse: $0.85 \times 0.15 = 0.1275$
   - Fire Outbreak: $0.75 \times 0.15 = 0.1125$
   - Flash Flood: $0.70 \times 0.15 = 0.1050$
   - Medical Emergency: $0.60 \times 0.15 = 0.0900$
   - Road Accident: $0.55 \times 0.15 = 0.0825$

4. **Weather Stress (10% max weight):**
   $$R_{\text{weather}} = \min\left(\frac{\text{rain\_mm}}{50.0}, 1.0\right) \times 0.05 + \max\left(0, \frac{10.0 - \text{visibility\_km}}{10.0}\right) \times 0.03 + \min\left(\frac{\text{wind\_kmh}}{60.0}, 1.0\right) \times 0.02$$

5. **Traffic Gridlock Vulnerability (8% max weight):**
   $$R_{\text{traffic}} = \text{traffic\_congestion} \times 0.08$$

6. **Road Infrastructure Criticality (7% max weight):**
   $$R_{\text{road}} = \begin{cases} 0.07 & \text{if road\_type} \in \{\text{'highway'}, \text{'bridge'}\} \\ 0.02 & \text{otherwise} \end{cases}$$

7. **Operational Window / Time-of-Day (3% max weight):**
   $$R_{\text{time}} = \begin{cases} 0.03 & \text{if hour} \in \{8, 9, 10, 17, 18, 19, 20\} \cup [22, 5] \\ 0.00 & \text{otherwise} \end{cases}$$

The score is clamped: $R_{\text{clamped}} = \max(0.02, \min(0.99, R))$.  
The binary classification label is:
$$\text{is\_high\_risk} = \begin{cases} 1 & \text{if } R_{\text{clamped}} \ge 0.55 \\ 0 & \text{otherwise} \end{cases}$$

---

## 4. Dataset Schema

| Column Name | Type | Allowed Range | Description |
| :--- | :--- | :--- | :--- |
| `incident_id` | String | Format: `INC-###-##` | Unique incident identifier |
| `tick` | Integer | $\ge 0$ | Simulation time tick (1 tick = 15 min) |
| `timestamp` | String | ISO 8601 | Simulated date and time |
| `incident_type` | Categorical | `road_accident`, `fire`, `flood`, `medical_emergency`, `building_collapse` | Category of emergency |
| `severity` | Integer | $1$ to $5$ | Reported severity level |
| `latitude` | Float | $28.40$ to $28.88$ | GPS Latitude within Delhi bounds |
| `longitude` | Float | $76.84$ to $77.35$ | GPS Longitude within Delhi bounds |
| `people_affected` | Integer | $0$ to $500$ | Estimated people at risk or injured |
| `road_type` | Categorical | `highway`, `arterial`, `local`, `bridge` | Physical corridor classification |
| `traffic_congestion`| Float | $0.0$ to $1.0$ | Real-time road congestion index |
| `temperature_c` | Float | $-10.0$ to $55.0$ | Ambient temperature in Celsius |
| `rain_mm` | Float | $\ge 0.0$ | Rainfall in mm/hour |
| `wind_kmh` | Float | $\ge 0.0$ | Wind speed in km/hour |
| `visibility_km` | Float | $0.1$ to $30.0$ | Horizontal visibility in km |
| `true_risk_score` | Float | $0.0$ to $1.0$ | Continuous ground-truth risk score |
| `is_high_risk` | Integer | $0$ or $1$ | Ground-truth binary classification target |

---

## 5. Transitioning to Real Public Data

In production municipal deployments, this simulation pipeline can be substituted with open public municipal feeds:

1. **Seattle Open Data (Real-Time 911 Calls):**
   - URL: [data.seattle.gov](https://data.seattle.gov)
   - Features: Timestamps, incident categories, geographic coordinates, responding battalion units.
2. **NYC Open Data (Incident-Level CAD Responses):**
   - URL: [opendata.cityofnewyork.us](https://opendata.cityofnewyork.us)
   - Features: Call received-to-arrival response times, verified severity, unit dispatches.
3. **UK Police Street-Level Crime and Emergency Data:**
   - URL: [data.police.uk](https://data.police.uk)
   - Features: Monthly geo-located incident records.
4. **OpenStreetMap Road Network Graph:**
   - Extracts realistic road class, speed limits, and bridge/tunnel vulnerability features.
