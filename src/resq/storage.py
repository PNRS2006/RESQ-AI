"""SQLite database engine managing incidents, AI recommendations, and human decision audit trails."""

from datetime import datetime, timezone
import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from resq.config import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


class ResQStorage:
    """Thread-safe SQLite storage engine for ResQ-AI."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with dict-like row access."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database schema if not present."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Incidents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    tick INTEGER,
                    timestamp TEXT,
                    incident_type TEXT,
                    severity INTEGER,
                    latitude REAL,
                    longitude REAL,
                    people_affected INTEGER,
                    road_type TEXT,
                    traffic_congestion REAL,
                    temperature_c REAL,
                    rain_mm REAL,
                    wind_kmh REAL,
                    visibility_km REAL,
                    risk_score REAL,
                    urgency_score REAL,
                    status TEXT DEFAULT 'PENDING'
                )
            """)

            # 2. Recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT,
                    recommended_unit_id TEXT,
                    unit_type TEXT,
                    distance_km REAL,
                    effective_speed_kmh REAL,
                    eta_minutes REAL,
                    plain_reason TEXT,
                    created_at TEXT,
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                )
            """)

            # 3. Decisions table (Human-in-the-loop audit log)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT,
                    commander_id TEXT,
                    decision TEXT,
                    assigned_unit_id TEXT,
                    comments TEXT,
                    decided_at TEXT,
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                )
            """)

            conn.commit()
            logger.info("Database initialized at %s", self.db_path)

    def save_incidents(self, incidents: List[Dict[str, Any]]) -> int:
        """Upsert incident records into database."""
        if not incidents:
            return 0
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            saved_count = 0
            for inc in incidents:
                cursor.execute("""
                    INSERT OR REPLACE INTO incidents (
                        incident_id, tick, timestamp, incident_type, severity,
                        latitude, longitude, people_affected, road_type,
                        traffic_congestion, temperature_c, rain_mm, wind_kmh,
                        visibility_km, risk_score, urgency_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(inc.get("incident_id")),
                    int(inc.get("tick", 0)),
                    str(inc.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    str(inc.get("incident_type", "road_accident")),
                    int(inc.get("severity", 3)),
                    float(inc.get("latitude", 28.61)),
                    float(inc.get("longitude", 77.20)),
                    int(inc.get("people_affected", 0)),
                    str(inc.get("road_type", "arterial")),
                    float(inc.get("traffic_congestion", 0.5)),
                    float(inc.get("temperature_c", 28.0)),
                    float(inc.get("rain_mm", 0.0)),
                    float(inc.get("wind_kmh", 10.0)),
                    float(inc.get("visibility_km", 10.0)),
                    float(inc.get("risk_score", inc.get("true_risk_score", 0.5))),
                    float(inc.get("urgency_score", 0.5)),
                    str(inc.get("status", "PENDING")),
                ))
                saved_count += 1
            conn.commit()
            return saved_count

    def get_incidents(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent incidents, optionally filtered by status."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM incidents WHERE status = ?
                    ORDER BY urgency_score DESC, severity DESC LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM incidents
                    ORDER BY urgency_score DESC, severity DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def save_recommendations(self, recs: List[Dict[str, Any]]) -> int:
        """Insert resource recommendations."""
        if not recs:
            return 0
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            count = 0
            for r in recs:
                cursor.execute("""
                    INSERT INTO recommendations (
                        incident_id, recommended_unit_id, unit_type,
                        distance_km, effective_speed_kmh, eta_minutes,
                        plain_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(r.get("incident_id")),
                    r.get("recommended_unit_id"),
                    r.get("unit_type"),
                    r.get("distance_km"),
                    r.get("effective_speed_kmh"),
                    r.get("estimated_eta_minutes", r.get("eta_minutes")),
                    r.get("plain_reason", ""),
                    str(r.get("created_at", datetime.now(timezone.utc).isoformat())),
                ))
                count += 1
            conn.commit()
            return count

    def get_recommendations(self, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve latest recommendations."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if incident_id:
                cursor.execute(
                    "SELECT * FROM recommendations WHERE incident_id = ? ORDER BY rec_id DESC",
                    (incident_id,)
                )
            else:
                cursor.execute("SELECT * FROM recommendations ORDER BY rec_id DESC LIMIT 100")
            return [dict(r) for r in cursor.fetchall()]

    def log_human_decision(
        self,
        incident_id: str,
        commander_id: str,
        decision: str,  # 'APPROVE', 'REJECT', 'OVERRIDE'
        assigned_unit_id: Optional[str] = None,
        comments: str = "",
    ) -> int:
        """Record human commander approval or rejection, and update incident status."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decisions (
                    incident_id, commander_id, decision, assigned_unit_id, comments, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (incident_id, commander_id, decision, assigned_unit_id, comments, timestamp))
            decision_id = cursor.lastrowid

            # Update incident state
            new_status = "APPROVED" if decision == "APPROVE" else "REJECTED"
            cursor.execute(
                "UPDATE incidents SET status = ? WHERE incident_id = ?",
                (new_status, incident_id)
            )
            conn.commit()
            return decision_id or 0

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch chronological audit log of commander decisions."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.*, i.incident_type, i.severity, i.urgency_score
                FROM decisions d
                LEFT JOIN incidents i ON d.incident_id = i.incident_id
                ORDER BY d.decision_id DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]


# Global storage singleton
storage = ResQStorage()
