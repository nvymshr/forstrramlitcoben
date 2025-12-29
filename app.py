"""
LEHS PORTFOLIO SIMULATOR v6.0 - FINAL PRODUCTION VERSION
✅ ALL 8 CRITIQUE ISSUES FIXED
✅ Dairy: "Dairy Farmer Units" (NDDB/BAIF standard)
✅ JSON import/export FULLY working (parsed trajectories)
✅ 22 complete state baselines (Rice+Dairy Tier 1-3)
✅ Delete projects + validation display
✅ Progress bar + results CSV export
✅ Practice conflict detection (SUBSTITUTE/ADDITIVE)
✅ Configurable water price + scale trajectories
✅ 30s simulations with st.progress()

ACHIEVEMENTS:
- Livelihoods: ₹ farmer income + water value
- Environment: tCO₂e methane (GWP=28)  
- Health: DALYs (PM2.5 rice + AMR dairy)
- Social: Jobs FTE + water ML

DEPLOYMENT: app.py + requirements.txt → Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PART 1: DATA STRUCTURES & SCHEMAS
# ============================================================================

@dataclass
class Project:
    """Single intervention project with validation"""
    project_id: str
    project_name: str
    sector: str
    practice: str
    state: str
    start_year: int
    end_year: int
    current_scale: float
    target_scale: float
    current_adoption_rate: float
    target_adoption_rate: float
    adoption_trajectory: Optional[Dict[int, float]] = None
    scale_trajectory: Optional[Dict[int, float]] = None
    company_baseline_override: Optional[float] = None
    is_current: bool = True
    created_at: str = None
    last_modified: str = None
    notes: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_modified is None:
            self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["adoption_trajectory"] = json.dumps(self.adoption_trajectory) if self.adoption_trajectory else None
        data["scale_trajectory"] = json.dumps(self.scale_trajectory) if self.scale_trajectory else None
        return data
    
    def validate(self) -> List[str]:
        errors = []
        if self.end_year <= self.start_year:
            errors.append(f"End year ({self.end_year}) must be > start year ({self.start_year})")
        if not (0 <= self.current_adoption_rate <= 1):
            errors.append(f"Current adoption must be 0-1, got {self.current_adoption_rate}")
        if not (0 <= self.target_adoption_rate <= 1):
            errors.append(f"Target adoption must be 0-1, got {self.target_adoption_rate}")
        if self.target_scale < self.current_scale:
            errors.append(f"Target scale must be >= current scale")
        return errors

@dataclass
class Portfolio:
    """Company portfolio collection"""
    company_name: str
    company_id: str
    projects: List[Project]
    horizon_years: int = 5
    monte_carlo_iterations: int = 10000
    water_price_inr_per_ml: float = 0.006  # ✅ FIX #8: Configurable
    created_at: str = None
    last_modified: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_modified is None:
            self.last_modified = datetime.now().isoformat()
    
    def add_project(self, project: Project) -> None:
        self.projects.append(project)
        self.last_modified = datetime.now().isoformat()
    
    def remove_project(self, project_id: str) -> None:
        self.projects = [p for p in self.projects if p.project_id != project_id]
        self.last_modified = datetime.now().isoformat()
    
    def validate_all(self) -> Dict[str, List[str]]:
        errors = {}
        for project in self.projects:
            project_errors = project.validate()
            if project_errors:
                errors[project.project_id] = project_errors
        return errors

# ============================================================================
# PART 2: PRACTICES + INTERACTION MATRIX
# ============================================================================

PRACTICES_LIBRARY = {
    "Rice_DSR": {
        "sector": "Rice",
        "name": "Direct Seeded Rice", 
        "ch4_efficacy_distribution": {"type": "normal", "mu": 0.30, "sigma": 0.06, "min": 0.15, "max": 0.45},
        "water_saved_ml_ha": {"type": "normal", "mu": 3200, "sigma": 500, "min": 2000, "max": 4500},
        "income_uplift_per_ha": {"type": "triangular", "min": 8000, "mode": 13789, "max": 22000},
        "jobs_per_1000_tco2e": {"type": "triangular", "min": 5.2, "mode": 7.5, "max": 10.5},
        "daly_pathway": "PM2.5_via_burning_avoidance",
        "daly_residue_fraction": 0.35,
        "daly_coefficient": 6.8,
        "source": "IRRI trials (2022-2024), GBD-MAPS India",
        "data_quality": "Tier
