from valve_database import load_valves_from_excel, add_valve_to_database, delete_valve_from_database
from valve import Valve
from scipy.interpolate import CubicSpline
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
import base64
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import requests
from PIL import Image
import traceback
import kaleido
import CoolProp.CoolProp as CP

# ========================
# CONSTANTS & UNIT CONVERSION
# ========================
# Conversion factors
BAR_TO_KPA = 100
KPA_TO_BAR = 0.01
C_TO_K = 273.15
F_TO_R = 459.67
PSI_TO_BAR = 0.0689476
PSIA_TO_BAR = 0.0689476  # Same as PSI for absolute pressure
INCH_TO_MM = 25.4
MM_TO_INCH = 0.0393701
INCH_TO_M = 0.0254
M_TO_INCH = 39.3701
FT_TO_M = 0.3048
G_CONST = 9.80665
MMHG_TO_BAR = 0.00133322
INHG_TO_BAR = 0.0338639
FT_H2O_TO_BAR = 0.0298907
M_H2O_TO_BAR = 0.0980665
ATM_TO_BAR = 1.01325
KGCM2_TO_BAR = 0.980665

# Velocity limits (m/s)
VELOCITY_LIMITS = {"liquid": 5, "gas": 15, "steam": 15}

# Conversion factor between Cv and Kv: Kv = Cv / 1.156
CV_TO_KV = 1.156  # Conversion factor: Cv = 1.156 * Kv

CONSTANTS = {
    "N1": {"gpm, psia": 1.00, "m³/h, bar": 0.865, "m³/h, kPa": 0.0865},
    "N2": {"mm": 0.00214, "inch": 890},
    "N4": {"mm": 76000, "inch": 17300},
    "N5": {"mm": 0.00241, "inch": 1000},
    "N6": {"kg/h, kPa, kg/m³": 2.73, "kg/h, bar, kg/m³": 27.3, "lb/h, psia, lb/ft³": 63.3},
    "N7": {"m³/h, kPa, K (standard)": 4.17, "m³/h, bar, K (standard)": 417, "scfh, psia, R": 1360},
    "N8": {"kg/h, kPa, K": 0.948, "kg/h, bar, K": 94.8, "lb/h, psia, R": 19.3},
    "N9": {"m³/h, kPa, K (standard)": 22.4, "m³/h, bar, K (standard)": 2240, "scfh, psia, R": 7320}
}

WATER_DENSITY_4C = 999.97
AIR_DENSITY_0C = 1.293

# Standard conditions for gas flow
STANDARD_TEMPERATURE = 15 + C_TO_K  # 15°C in Kelvin
STANDARD_PRESSURE = 1.01325  # bar

# ========================
# COMPREHENSIVE UNIT CONVERSION FUNCTIONS
# ========================
class UnitConverter:
    """Class to handle all unit conversions in the application"""
    
    @staticmethod
    def convert_pressure(value, from_unit, to_unit="bar"):
        """
        Convert pressure between different units.
        
        Supported units:
        - bar, kPa, MPa, psi, psia, mmHg, inHg, ftH2O, mH2O, atm, kg/cm²
        
        Parameters:
        - value: Pressure value
        - from_unit: Source unit
        - to_unit: Target unit (default: bar for internal calculations)
        
        Returns:
        - Converted pressure value
        """
        if value is None or math.isnan(value):
            return 0
        
        if from_unit == to_unit:
            return value
        
        # Convert to bar first (standard internal unit)
        pressure_bar = value
        
        # Convert from source unit to bar
        if from_unit == "bar" or from_unit == "bar a":
            pressure_bar = value
        elif from_unit == "kPa":
            pressure_bar = value * KPA_TO_BAR
        elif from_unit == "MPa":
            pressure_bar = value * 10  # 1 MPa = 10 bar
        elif from_unit in ["psi", "psia"]:
            pressure_bar = value * PSI_TO_BAR
        elif from_unit == "mmHg":
            pressure_bar = value * MMHG_TO_BAR
        elif from_unit == "inHg":
            pressure_bar = value * INHG_TO_BAR
        elif from_unit == "ftH2O":
            pressure_bar = value * FT_H2O_TO_BAR
        elif from_unit == "mH2O":
            pressure_bar = value * M_H2O_TO_BAR
        elif from_unit == "atm":
            pressure_bar = value * ATM_TO_BAR
        elif from_unit == "kg/cm²":
            pressure_bar = value * KGCM2_TO_BAR
        else:
            # Default to bar
            pressure_bar = value
        
        # Convert from bar to target unit
        if to_unit == "bar" or to_unit == "bar a":
            return pressure_bar
        elif to_unit == "kPa":
            return pressure_bar / KPA_TO_BAR
        elif to_unit == "MPa":
            return pressure_bar / 10
        elif to_unit in ["psi", "psia"]:
            return pressure_bar / PSI_TO_BAR
        elif to_unit == "mmHg":
            return pressure_bar / MMHG_TO_BAR
        elif to_unit == "inHg":
            return pressure_bar / INHG_TO_BAR
        elif to_unit == "ftH2O":
            return pressure_bar / FT_H2O_TO_BAR
        elif to_unit == "mH2O":
            return pressure_bar / M_H2O_TO_BAR
        elif to_unit == "atm":
            return pressure_bar / ATM_TO_BAR
        elif to_unit == "kg/cm²":
            return pressure_bar / KGCM2_TO_BAR
        else:
            return pressure_bar
    
    @staticmethod
    def convert_temperature(value, from_unit, to_unit="°C"):
        """
        Convert temperature between different units.
        
        Supported units:
        - °C, K, °F, °R
        
        Parameters:
        - value: Temperature value
        - from_unit: Source unit
        - to_unit: Target unit (default: °C for internal calculations)
        
        Returns:
        - Converted temperature value
        """
        if value is None or math.isnan(value):
            return 0
        
        if from_unit == to_unit:
            return value
        
        # Convert to Celsius first (standard internal unit)
        temp_c = value
        
        # Convert from source unit to Celsius
        if from_unit == "°C":
            temp_c = value
        elif from_unit == "K":
            temp_c = value - C_TO_K
        elif from_unit == "°F":
            temp_c = (value - 32) * 5/9
        elif from_unit == "°R":  # Rankine
            temp_c = (value - 491.67) * 5/9
        else:
            # Default to Celsius
            temp_c = value
        
        # Convert from Celsius to target unit
        if to_unit == "°C":
            return temp_c
        elif to_unit == "K":
            return temp_c + C_TO_K
        elif to_unit == "°F":
            return temp_c * 9/5 + 32
        elif to_unit == "°R":
            return (temp_c + C_TO_K) * 9/5
        else:
            return temp_c
    
    @staticmethod
    def convert_length(value, from_unit, to_unit="inch"):
        """
        Convert length/diameter between different units.
        
        Supported units:
        - inch, mm, cm, m, ft
        
        Parameters:
        - value: Length value
        - from_unit: Source unit
        - to_unit: Target unit (default: inch for internal calculations)
        
        Returns:
        - Converted length value
        """
        if value is None or math.isnan(value):
            return 0
        
        if from_unit == to_unit:
            return value
        
        # Convert to inches first (standard internal unit for valve/pipe diameters)
        length_inch = value
        
        # Convert from source unit to inches
        if from_unit == "inch":
            length_inch = value
        elif from_unit == "mm":
            length_inch = value * MM_TO_INCH
        elif from_unit == "cm":
            length_inch = value * MM_TO_INCH * 10
        elif from_unit == "m":
            length_inch = value * M_TO_INCH
        elif from_unit == "ft":
            length_inch = value * 12
        else:
            # Default to inches
            length_inch = value
        
        # Convert from inches to target unit
        if to_unit == "inch":
            return length_inch
        elif to_unit == "mm":
            return length_inch / MM_TO_INCH
        elif to_unit == "cm":
            return length_inch / (MM_TO_INCH * 10)
        elif to_unit == "m":
            return length_inch / M_TO_INCH
        elif to_unit == "ft":
            return length_inch / 12
        else:
            return length_inch
    
    @staticmethod
    def convert_flow_rate(value, from_unit, to_unit, fluid_type, temp_c=None, pressure_bar=None):
        """
        Convert flow rate between different units with support for fluid type.
        
        Supported units:
        Liquids: m³/h, L/min, L/s, US gpm, Imp gpm, bbl/h
        Gases: std m³/h, scfm, scfh, Nm³/h, MMSCFD, L/min
        Steam: kg/h, lb/h, t/h, kg/s
        
        Parameters:
        - value: Flow rate value
        - from_unit: Source unit
        - to_unit: Target unit
        - fluid_type: "liquid", "gas", or "steam"
        - temp_c: Temperature in °C (for gas conversions)
        - pressure_bar: Pressure in bar (for gas conversions)
        
        Returns:
        - Converted flow rate value
        """
        if value is None or math.isnan(value) or value == 0:
            return 0
        
        if from_unit == to_unit:
            return value
        
        # Define conversion factors to base units
        FLOW_RATE_UNITS = {
            "liquid": {
                "m³/h": 1.0,
                "L/min": 0.06,  # 1 L/min = 0.06 m³/h
                "L/s": 3.6,     # 1 L/s = 3.6 m³/h
                "US gpm": 0.227124,  # 1 US gpm = 0.227124 m³/h
                "Imp gpm": 0.272765,  # 1 Imperial gpm = 0.272765 m³/h
                "bbl/h": 0.158987,   # 1 bbl/h = 0.158987 m³/h
            },
            "gas": {
                "std m³/h": 1.0,
                "scfm": 1.699,     # 1 scfm = 1.699 std m³/h
                "scfh": 0.0283168,  # 1 scfh = 0.0283168 std m³/h
                "Nm³/h": 1.0,       # Nm³/h is equivalent to std m³/h
                "MMSCFD": 1177.17,  # 1 MMSCFD = 1177.17 std m³/h
                "L/min": 0.06,      # 1 L/min = 0.06 std m³/h (assuming standard conditions)
            },
            "steam": {
                "kg/h": 1.0,
                "lb/h": 0.453592,   # 1 lb/h = 0.453592 kg/h
                "t/h": 1000.0,      # 1 t/h = 1000 kg/h
                "kg/s": 3600.0,     # 1 kg/s = 3600 kg/h
            }
        }
        
        # Get conversion factors for the fluid type
        if fluid_type in FLOW_RATE_UNITS:
            units_dict = FLOW_RATE_UNITS[fluid_type]
            
            # Convert to base unit first
            if from_unit in units_dict:
                value_in_base = value * units_dict[from_unit]
            else:
                # If unit not found, assume it's already in base unit
                value_in_base = value
            
            # Convert from base unit to target unit
            if to_unit in units_dict:
                if units_dict[to_unit] != 0:
                    return value_in_base / units_dict[to_unit]
        
        return value
    
    @staticmethod
    def convert_density(value, from_unit, to_unit="kg/m³"):
        """
        Convert density between different units.
        
        Supported units:
        - kg/m³, g/cm³, lb/ft³, lb/gal (US), SG (specific gravity)
        
        Parameters:
        - value: Density value
        - from_unit: Source unit
        - to_unit: Target unit
        
        Returns:
        - Converted density value
        """
        if value is None or math.isnan(value):
            return 0
        
        if from_unit == to_unit:
            return value
        
        # Convert to kg/m³ first (standard internal unit)
        density_kg_m3 = value
        
        # Convert from source unit to kg/m³
        if from_unit == "kg/m³":
            density_kg_m3 = value
        elif from_unit == "g/cm³":
            density_kg_m3 = value * 1000
        elif from_unit == "lb/ft³":
            density_kg_m3 = value * 16.0185
        elif from_unit == "lb/gal (US)":
            density_kg_m3 = value * 119.826
        elif from_unit == "SG":  # Specific gravity (relative to water)
            density_kg_m3 = value * 1000  # Water density approx 1000 kg/m³
        else:
            # Default to kg/m³
            density_kg_m3 = value
        
        # Convert from kg/m³ to target unit
        if to_unit == "kg/m³":
            return density_kg_m3
        elif to_unit == "g/cm³":
            return density_kg_m3 / 1000
        elif to_unit == "lb/ft³":
            return density_kg_m3 / 16.0185
        elif to_unit == "lb/gal (US)":
            return density_kg_m3 / 119.826
        elif to_unit == "SG":
            return density_kg_m3 / 1000
        else:
            return density_kg_m3
    
    @staticmethod
    def convert_viscosity(value, from_unit, to_unit="cSt"):
        """
        Convert viscosity between different units.
        
        Supported units:
        - cSt (centistokes), m²/s, ft²/s, SSF, SSU
        
        Parameters:
        - value: Viscosity value
        - from_unit: Source unit
        - to_unit: Target unit
        
        Returns:
        - Converted viscosity value
        """
        if value is None or math.isnan(value):
            return 1.0
        
        if from_unit == to_unit:
            return value
        
        # Convert to cSt first (standard internal unit)
        viscosity_cst = value
        
        # Convert from source unit to cSt
        if from_unit == "cSt":
            viscosity_cst = value
        elif from_unit == "m²/s":
            viscosity_cst = value * 1e6
        elif from_unit == "ft²/s":
            viscosity_cst = value * 92903.04
        elif from_unit == "SSF":  # Saybolt Seconds Furol
            # Approximate conversion: cSt ≈ 0.22 * SSF - 180/SSF
            viscosity_cst = 0.22 * value - 180/value if value > 0 else 1.0
        elif from_unit == "SSU":  # Saybolt Seconds Universal
            # Approximate conversion: cSt ≈ 0.22 * SSU - 135/SSU
            viscosity_cst = 0.22 * value - 135/value if value > 0 else 1.0
        else:
            # Default to cSt
            viscosity_cst = value
        
        # Convert from cSt to target unit
        if to_unit == "cSt":
            return viscosity_cst
        elif to_unit == "m²/s":
            return viscosity_cst / 1e6
        elif to_unit == "ft²/s":
            return viscosity_cst / 92903.04
        elif to_unit == "SSF":
            # Approximate inverse conversion
            return (viscosity_cst + math.sqrt(viscosity_cst**2 + 4*0.22*180)) / (2*0.22)
        elif to_unit == "SSU":
            # Approximate inverse conversion
            return (viscosity_cst + math.sqrt(viscosity_cst**2 + 4*0.22*135)) / (2*0.22)
        else:
            return viscosity_cst
    
    @staticmethod
    def format_unit_display(value, unit, decimals=2):
        """
        Format a value with its unit for display.
        
        Parameters:
        - value: The value to format
        - unit: The unit
        - decimals: Number of decimal places
        
        Returns:
        - Formatted string
        """
        if value is None or math.isnan(value):
            return "N/A"
        
        # Format based on magnitude
        if abs(value) >= 1000:
            formatted = f"{value:,.0f}"
        elif abs(value) >= 100:
            formatted = f"{value:,.1f}"
        elif abs(value) >= 10:
            formatted = f"{value:,.2f}"
        elif abs(value) >= 1:
            formatted = f"{value:,.3f}"
        elif abs(value) >= 0.1:
            formatted = f"{value:,.4f}"
        else:
            formatted = f"{value:,.6f}"
        
        # Remove trailing zeros after decimal point
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        
        return f"{formatted} {unit}"

# ========================
# ENHANCED CALCULATION RECORD CLASS
# ========================
class CalculationRecord:
    def __init__(self, scenario_name, fluid_type):
        self.scenario_name = scenario_name
        self.fluid_type = fluid_type
        self.steps = []
        self.formulas = []
        self.results = {}
        self.assumptions = []
        self.warnings = []
        self.iterations = []
        self.iteration_count = 0
        self.input_parameters = {}
        self.intermediate_values = {}
        self.final_values = {}
        self.fp_details = {}  # New: Store Fp calculation details
        self.input_units = {}  # Store original input units
        
    def add_step(self, step_number, description, formula, result, unit="", details=None):
        """Add a calculation step to the record with details"""
        step_data = {
            'step': step_number,
            'description': description,
            'formula': formula,
            'result': result,
            'unit': unit,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'details': details if details else {}
        }
        self.steps.append(step_data)
        
    def add_formula(self, name, formula, explanation, variables=None):
        """Add a formula used in calculation"""
        self.formulas.append({
            'name': name,
            'formula': formula,
            'explanation': explanation,
            'variables': variables if variables else {}
        })
        
    def add_assumption(self, assumption, reason, details=None):
        """Add an assumption made during calculation"""
        self.assumptions.append({
            'assumption': assumption,
            'reason': reason,
            'details': details if details else {}
        })
        
    def add_warning(self, warning, severity="info", details=None):
        """Add a warning or note"""
        self.warnings.append({
            'warning': warning,
            'severity': severity,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'details': details if details else {}
        })
        
    def add_iteration(self, iteration_num, data):
        """Add iteration data"""
        self.iterations.append({
            'iteration': iteration_num,
            'data': data,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        self.iteration_count = iteration_num
        
    def set_input_parameter(self, key, value, unit="", original_value=None, original_unit=None):
        """Set an input parameter with original units"""
        self.input_parameters[key] = {
            'value': value, 
            'unit': unit,
            'original_value': original_value if original_value is not None else value,
            'original_unit': original_unit if original_unit is not None else unit
        }
        
    def set_intermediate_value(self, key, value, unit="", description=""):
        """Set an intermediate calculation value"""
        self.intermediate_values[key] = {
            'value': value, 
            'unit': unit, 
            'description': description,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
    def set_result(self, key, value, unit="", description=""):
        """Set a final result value"""
        self.results[key] = {
            'value': value, 
            'unit': unit,
            'description': description
        }
    
    def set_fp_details(self, details):
        """Store Fp calculation details"""
        self.fp_details = details
        
    def set_input_unit(self, param_name, unit):
        """Store original input unit for a parameter"""
        self.input_units[param_name] = unit
        
    def get_input_unit(self, param_name):
        """Get original input unit for a parameter"""
        return self.input_units.get(param_name, "")
    
    def get_summary(self):
        """Get a summary of the calculation"""
        return {
            'scenario': self.scenario_name,
            'fluid_type': self.fluid_type,
            'total_steps': len(self.steps),
            'total_formulas': len(self.formulas),
            'total_assumptions': len(self.assumptions),
            'total_warnings': len(self.warnings),
            'iteration_count': self.iteration_count,
            'steps': self.steps,
            'formulas': self.formulas,
            'assumptions': self.assumptions,
            'warnings': self.warnings,
            'iterations': self.iterations,
            'input_parameters': self.input_parameters,
            'intermediate_values': self.intermediate_values,
            'results': self.results,
            'fp_details': self.fp_details,
            'input_units': self.input_units
        }
        
    def generate_detailed_report(self):
        """Generate a detailed text report of the calculation"""
        report = []
        report.append("=" * 80)
        report.append(f"CALCULATION REPORT: {self.scenario_name}")
        report.append("=" * 80)
        report.append(f"Fluid Type: {self.fluid_type.upper()}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Input Parameters with original units
        report.append("INPUT PARAMETERS:")
        report.append("-" * 40)
        for key, data in self.input_parameters.items():
            if 'original_value' in data and 'original_unit' in data:
                report.append(f"  {key}: {data['original_value']} {data['original_unit']} (converted to: {data['value']} {data['unit']})")
            else:
                report.append(f"  {key}: {data['value']} {data['unit']}")
        report.append("")
        
        # Formulas Used
        report.append("FORMULAS USED:")
        report.append("-" * 40)
        for formula in self.formulas:
            report.append(f"  {formula['name']}:")
            report.append(f"    Formula: {formula['formula']}")
            report.append(f"    Explanation: {formula['explanation']}")
            if formula.get('variables'):
                report.append(f"    Variables: {formula['variables']}")
            report.append("")
            
        # Fp Calculation Details (if available)
        if self.fp_details:
            report.append("PIPING GEOMETRY FACTOR (Fp) CALCULATION:")
            report.append("-" * 40)
            report.append(f"  Valve diameter: {self.fp_details.get('valve_d_inch', 'N/A')} inch")
            report.append(f"  Inlet pipe diameter: {self.fp_details.get('pipe_d_in_inch', 'N/A')} inch")
            report.append(f"  Outlet pipe diameter: {self.fp_details.get('pipe_d_out_inch', 'N/A')} inch")
            report.append(f"  Cv used for calculation: {self.fp_details.get('cv_op', 'N/A'):.1f}")
            report.append(f"  d_ratio_in (valve/inlet): {self.fp_details.get('d_ratio_in', 'N/A'):.4f}")
            report.append(f"  d_ratio_out (valve/outlet): {self.fp_details.get('d_ratio_out', 'N/A'):.4f}")
            report.append(f"  K1 (inlet reducer): {self.fp_details.get('K1', 'N/A'):.4f}")
            report.append(f"  K2 (outlet reducer): {self.fp_details.get('K2', 'N/A'):.4f}")
            report.append(f"  KB1 (inlet Bernoulli): {self.fp_details.get('KB1', 'N/A'):.4f}")
            report.append(f"  KB2 (outlet Bernoulli): {self.fp_details.get('KB2', 'N/A'):.4f}")
            report.append(f"  ΣK = K1 + K2 + KB1 - KB2: {self.fp_details.get('sumK', 'N/A'):.4f}")
            report.append(f"  N2 constant: {self.fp_details.get('N2', 'N/A')}")
            report.append(f"  Term = 1 + (ΣK/N2) * (Cv/d²)²: {self.fp_details.get('term', 'N/A'):.4f}")
            report.append(f"  Fp = 1 / √(Term): {self.fp_details.get('Fp', 'N/A'):.4f}")
            report.append("")
            
        # Calculation Steps
        report.append("CALCULATION STEPS:")
        report.append("-" * 40)
        for step in self.steps:
            report.append(f"  Step {step['step']}: {step['description']}")
            report.append(f"    Formula: {step['formula']}")
            report.append(f"    Result: {step['result']} {step['unit']}")
            if step.get('details'):
                for detail_key, detail_value in step['details'].items():
                    report.append(f"      {detail_key}: {detail_value}")
            report.append(f"    Time: {step['timestamp']}")
            report.append("")
            
        # Iterations
        if self.iterations:
            report.append("ITERATION HISTORY:")
            report.append("-" * 40)
            for iteration in self.iterations:
                report.append(f"  Iteration {iteration['iteration']}:")
                for key, value in iteration['data'].items():
                    if isinstance(value, (int, float)):
                        report.append(f"    {key}: {value}")
                    elif isinstance(value, dict):
                        report.append(f"    {key}:")
                        for subkey, subvalue in value.items():
                            report.append(f"      {subkey}: {subvalue}")
                report.append(f"    Time: {iteration['timestamp']}")
                report.append("")
                
        # Intermediate Values
        if self.intermediate_values:
            report.append("INTERMEDIATE VALUES:")
            report.append("-" * 40)
            for key, data in self.intermediate_values.items():
                report.append(f"  {key}: {data['value']} {data['unit']}")
                if data['description']:
                    report.append(f"    Description: {data['description']}")
                report.append(f"    Calculated at: {data['timestamp']}")
            report.append("")
            
        # Assumptions
        if self.assumptions:
            report.append("ASSUMPTIONS:")
            report.append("-" * 40)
            for assumption in self.assumptions:
                report.append(f"  • {assumption['assumption']}")
                report.append(f"    Reason: {assumption['reason']}")
                if assumption.get('details'):
                    for detail_key, detail_value in assumption['details'].items():
                        report.append(f"    {detail_key}: {detail_value}")
                report.append("")
                
        # Warnings
        if self.warnings:
            report.append("WARNINGS AND NOTES:")
            report.append("-" * 40)
            for warning in self.warnings:
                report.append(f"  [{warning['severity'].upper()}] {warning['warning']}")
                if warning.get('details'):
                    for detail_key, detail_value in warning['details'].items():
                        report.append(f"    {detail_key}: {detail_value}")
                report.append(f"    Time: {warning['timestamp']}")
                report.append("")
                
        # Final Results
        report.append("FINAL RESULTS:")
        report.append("-" * 40)
        for key, data in self.results.items():
            report.append(f"  {key}: {data['value']} {data['unit']}")
            if data['description']:
                report.append(f"    Description: {data['description']}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF CALCULATION REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_formatted_steps(self):
        """Get formatted steps for display"""
        formatted = []
        for step in self.steps:
            step_text = f"**Step {step['step']}:** {step['description']}\n"
            step_text += f"  *Formula:* `{step['formula']}`\n"
            step_text += f"  *Result:* `{step['result']}` {step['unit']}\n"
            if step.get('details'):
                for detail_key, detail_value in step['details'].items():
                    step_text += f"  *{detail_key}:* `{detail_value}`\n"
            formatted.append(step_text)
        return formatted
    
    def get_formatted_iterations(self):
        """Get formatted iterations for display"""
        formatted = []
        for iteration in self.iterations:
            iter_text = f"**Iteration {iteration['iteration']}:**\n"
            for key, value in iteration['data'].items():
                if isinstance(value, (int, float)):
                    iter_text += f"  *{key}:* `{value}`\n"
                elif isinstance(value, dict):
                    iter_text += f"  *{key}:*\n"
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            iter_text += f"    *{subkey}:* `{subvalue}`\n"
                        else:
                            iter_text += f"    *{subkey}:* {subvalue}\n"
            formatted.append(iter_text)
        return formatted

# ========================
# CV TO KV CONVERSION FUNCTIONS
# ========================
def cv_to_kv(cv_value):
    """Convert Cv to Kv"""
    if cv_value is None:
        return 0
    
    # Handle numpy arrays
    if hasattr(cv_value, '__iter__'):
        try:
            import numpy as np
            if isinstance(cv_value, np.ndarray):
                # Create output array
                result = np.zeros_like(cv_value, dtype=np.float64)
                # Only process positive values
                mask = cv_value > 0
                result[mask] = cv_value[mask] / CV_TO_KV
                return result
        except:
            pass
    
    # Handle scalar or other iterables
    try:
        if hasattr(cv_value, '__len__') and not isinstance(cv_value, (str, bytes)):
            return [cv_to_kv(x) for x in cv_value]
    except:
        pass
    
    # Handle scalar
    if cv_value <= 0:
        return 0
    return cv_value / CV_TO_KV

def kv_to_cv(kv_value):
    """Convert Kv to Cv"""
    if kv_value is None:
        return 0
    
    # Handle numpy arrays
    if hasattr(kv_value, '__iter__'):
        try:
            import numpy as np
            if isinstance(kv_value, np.ndarray):
                # Create output array
                result = np.zeros_like(kv_value, dtype=np.float64)
                # Only process positive values
                mask = kv_value > 0
                result[mask] = kv_value[mask] * CV_TO_KV
                return result
        except:
            pass
    
    # Handle scalar or other iterables
    try:
        if hasattr(kv_value, '__len__') and not isinstance(kv_value, (str, bytes)):
            return [kv_to_cv(x) for x in kv_value]
    except:
        pass
    
    # Handle scalar
    if kv_value <= 0:
        return 0
    return kv_value * CV_TO_KV

# ========================
# FLUID LIBRARY
# ========================
FLUID_LIBRARY = {
    "Water": {
        "type": "liquid",
        "coolprop_name": "Water",
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Water", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Water", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Water') / 1e5,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "air": {
        "type": "gas",
        "coolprop_name": "air",
        "sg": 1.0,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("air", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("air", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("air", t, p)
    },
    "Methane": {
        "type": "gas",
        "coolprop_name": "Methane",
        "sg": 0.554,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Methane", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Methane", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Methane", t, p)
    },
    "Steam": {
        "type": "steam",
        "coolprop_name": "Water",
        "sg": None,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Water", t, p),
        "z_func": None,  # Steam doesn't use Z in the same way
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "CarbonDioxide": {
        "type": "gas",
        "coolprop_name": "CarbonDioxide",
        "sg": 1.52,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("CarbonDioxide", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("CarbonDioxide", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("CarbonDioxide", t, p)
    },
    "Ammonia": {
        "type": "gas",
        "coolprop_name": "Ammonia",
        "sg": 0.59,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Ammonia", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Ammonia", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Ammonia", t, p)
    },
    "Hydrogen": {
        "type": "gas",
        "coolprop_name": "Hydrogen",
        "sg": 0.0696,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Hydrogen", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Hydrogen", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Hydrogen", t, p)
    },
    "20% Hydrogen + 80% Methane": {
        "type": "gas",
        "coolprop_name": None,
        "sg": 0.455,  # Calculated as (0.2*0.0696 + 0.8*0.554)
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane"),
        "z_func": lambda t, p: calculate_compressibility_factor_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane"),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane")
    },
    "Natural Gas": {
        "type": "gas",
        "coolprop_name": "Methane",  # Using methane as approximation for natural gas
        "sg": 0.6,  # Typical specific gravity for natural gas (range: 0.55-0.75)
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Methane", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Methane", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Methane", t, p)
    },
    "Octane": {
        "type": "liquid",
        "coolprop_name": "Octane",
        "sg": 0.74,
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Octane", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Octane", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Octane') / 1e5,
        "rho_func": lambda t, p: calculate_density("Octane", t, p)
    },
    "JP8": {
        "type": "liquid",
        "coolprop_name": "Decane",  # Using Decane as approximation for JP8
        "sg": 0.81,  # Typical specific gravity for JP8 (0.78-0.84)
        "visc_func": lambda t, p: calculate_kinematic_viscosity_jet_fuel(t, p, "JP8"),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure_jet_fuel(t, p, "JP8"),
        "pc_func": lambda: 23.0,  # Approximate critical pressure for JP8 in bar
        "rho_func": lambda t, p: calculate_density_jet_fuel(t, p, "JP8")
    },
    "JET A1": {
        "type": "liquid",
        "coolprop_name": "Decane",  # Using Decane as approximation for Jet A1
        "sg": 0.80,  # Typical specific gravity for Jet A1 (0.78-0.84)
        "visc_func": lambda t, p: calculate_kinematic_viscosity_jet_fuel(t, p, "JET A1"),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure_jet_fuel(t, p, "JET A1"),
        "pc_func": lambda: 22.0,  # Approximate critical pressure for Jet A1 in bar
        "rho_func": lambda t, p: calculate_density_jet_fuel(t, p, "JET A1")
    },
    "Propane": {
        "type": "liquid",
        "coolprop_name": "Propane",
        "sg": 0.507,  # Typical specific gravity at 15°C (density ~507 kg/m³)
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Propane", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Propane", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Propane') / 1e5,  # Critical pressure in bar
        "rho_func": lambda t, p: calculate_density("Propane", t, p)
    }
}

# ========================
# FLUID PROPERTY FUNCTIONS
# ========================
def calculate_kinematic_viscosity_jet_fuel(temp_c: float, press_bar: float, fuel_type: str) -> float:
    """
    Calculate kinematic viscosity for jet fuels (JP8, JET A1)
    Uses empirical correlations for jet fuels
    """
    try:
        # Try CoolProp first if available
        if fuel_type == "JP8":
            # Use Decane as approximation
            return calculate_kinematic_viscosity("Decane", temp_c, press_bar)
        elif fuel_type == "JET A1":
            # Use Decane as approximation
            return calculate_kinematic_viscosity("Decane", temp_c, press_bar)
    except:
        pass
    
    # Empirical correlation for jet fuel viscosity (cSt)
    # Viscosity decreases with temperature
    if fuel_type == "JP8":
        # JP8 viscosity correlation: ~1.8 cSt at 20°C, ~1.0 cSt at 40°C
        base_visc = 1.8
    else:  # JET A1
        # JET A1 viscosity correlation: ~1.5 cSt at 20°C, ~0.9 cSt at 40°C
        base_visc = 1.5
    
    # Temperature correction (simplified)
    temp_factor = math.exp(-0.03 * (temp_c - 20))
    viscosity = base_visc * temp_factor
    
    # Ensure reasonable bounds
    return max(0.5, min(viscosity, 10.0))

def calculate_vapor_pressure_jet_fuel(temp_c: float, press_bar: float, fuel_type: str) -> float:
    """
    Calculate vapor pressure for jet fuels (JP8, JET A1)
    Jet fuels have very low vapor pressures at normal temperatures
    """
    try:
        if fuel_type == "JP8":
            # Use Decane as approximation
            return calculate_vapor_pressure("Decane", temp_c, press_bar)
        elif fuel_type == "JET A1":
            # Use Decane as approximation
            return calculate_vapor_pressure("Decane", temp_c, press_bar)
    except:
        pass
    
    # Jet fuels have very low vapor pressures
    # Empirical approximation: increases exponentially with temperature
    if temp_c <= 0:
        return 0.0001
    elif temp_c <= 100:
        # Very low vapor pressure at normal temperatures
        return 0.001 * math.exp(0.05 * temp_c)
    else:
        # Higher vapor pressure at elevated temperatures
        return 0.1 * math.exp(0.03 * (temp_c - 100))

def calculate_density_jet_fuel(temp_c: float, press_bar: float, fuel_type: str) -> float:
    """
    Calculate density for jet fuels (JP8, JET A1)
    """
    try:
        if fuel_type == "JP8":
            # Use Decane as approximation
            return calculate_density("Decane", temp_c, press_bar)
        elif fuel_type == "JET A1":
            # Use Decane as approximation
            return calculate_density("Decane", temp_c, press_bar)
    except:
        pass
    
    # Base density at 15°C (kg/m³)
    if fuel_type == "JP8":
        base_density = 810.0  # kg/m³ at 15°C
    else:  # JET A1
        base_density = 800.0  # kg/m³ at 15°C
    
    # Temperature correction (typical for hydrocarbons: ~0.7 kg/m³ per °C)
    temp_correction = 0.7 * (temp_c - 15)
    density = base_density - temp_correction
    
    # Pressure correction (small for liquids)
    pressure_correction = 0.04 * (press_bar - 1)
    density += pressure_correction
    
    return max(700.0, min(density, 850.0))

def calculate_compressibility_factor(fluid: str, temp_c: float, press_bar: float) -> float:
    """Calculate compressibility factor Z using CoolProp"""
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            Z = CP.PropsSI('Z', 'T', T, 'P', P, fluid_name)
            return Z
    except:
        pass
    return 1.0

def calculate_compressibility_factor_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    """Calculate compressibility factor for gas mixtures"""
    try:
        Z1 = calculate_compressibility_factor(fluid1, temp_c, press_bar)
        Z2 = calculate_compressibility_factor(fluid2, temp_c, press_bar)
        return frac1 * Z1 + frac2 * Z2
    except:
        pass
    return 1.0

def calculate_vapor_pressure(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            pv = CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name) / 1e5
            return pv
    except:
        pass
    return 0.0

def calculate_density(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            return rho
    except:
        pass
    if fluid == "water":
        return WATER_DENSITY_4C
    elif fluid == "air":
        return (press_bar * 1e5) * 28.97 / (8.314462 * (temp_c + C_TO_K))
    return 1000

def calculate_density_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    try:
        T = temp_c + C_TO_K
        P = press_bar * 1e5
        rho1 = calculate_density(fluid1, temp_c, press_bar)
        rho2 = calculate_density(fluid2, temp_c, press_bar)
        return frac1 * rho1 + frac2 * rho2
    except:
        pass
    return 1.0

def calculate_kinematic_viscosity(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            mu = CP.PropsSI('VISCOSITY', 'T', T, 'P', P, fluid_name)
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            nu = mu / rho * 1e6
            return nu
    except:
        pass
    if fluid.lower() == "water":
        return 1.79 / (1 + 0.0337 * temp_c + 0.00022 * temp_c**2)
    elif fluid.lower() == "propane":
        # Viscosity of liquid propane decreases with temperature
        # At 20°C: ~0.2 cSt, at -40°C: ~1.0 cSt
        if temp_c <= -40:
            return 1.0
        elif temp_c >= 50:
            return 0.1
        else:
            # Exponential decrease with temperature
            return 1.0 * math.exp(-0.03 * (temp_c + 40))
    elif fluid.lower() == "octane":
        return 0.6 * math.exp(-0.04 * (temp_c - 20))
    return 1.0

def calculate_specific_heat_ratio(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            Cp = CP.PropsSI('CPMASS', 'T', T, 'P', P, fluid_name)
            Cv = CP.PropsSI('CVMASS', 'T', T, 'P', P, fluid_name)
            return Cp / Cv
    except:
        pass
    if fluid.lower() == "air":
        return 1.4 - 0.0001 * temp_c
    elif fluid.lower() == "methane":
        return 1.31 - 0.00008 * temp_c
    elif fluid.lower() == "steam":
        return 1.33 - 0.0001 * temp_c
    elif fluid.lower() == "carbondioxide":
        return 1.28 - 0.00005 * temp_c
    elif fluid.lower() == "ammonia":
        return 1.32 - 0.00007 * temp_c
    elif fluid.lower() == "hydrogen":
        return 1.41 - 0.0001 * temp_c
    elif fluid.lower() == "propane":
        # Specific heat ratio for propane gas (if used as gas)
        return 1.13 - 0.0001 * temp_c
    return 1.4

def calculate_specific_heat_ratio_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    try:
        k1 = calculate_specific_heat_ratio(fluid1, temp_c, press_bar)
        k2 = calculate_specific_heat_ratio(fluid2, temp_c, press_bar)
        return frac1 * k1 + frac2 * k2
    except:
        pass
    return 1.4

def calculate_ff(pv: float, pc: float) -> float:
    if pc <= 0:
        return 0.96
    return 0.96 - 0.28 * math.sqrt(pv / pc)

# ========================
# VALVE DATABASE
# ========================
VALVE_DATABASE = load_valves_from_excel()

# ========================
# VALVE MODELS (static mapping)
# ========================
VALVE_MODELS = {
    "1.0\" E33": "https://example.com/models/0_5E31.glb",
    "2.0\" E33": "https://example.com/models/1E31.glb",
    "4.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/obje8e43.glb",
    "8.0\" E33": "https://github.com/gurkan-maker/proje/raw/94a24395c07603d1f94114fb1d12672a5dcdc50b/8e43.glb",
    "8.0\" E43": "https://h6zcfagabwtbjri5.public.blob.vercel-storage.com/8e43-wg0eEqrfslcYlhRqoQhYcNbNpSDBqK.glb",
    "12.0\" E33": "https://example.com/models/3E32.glb",
    "16.0\" E33": "https://example.com/models/4E32.glb",
    "20.0\" E33": "https://example.com/models/6E32.glb",
    "30.0\" E33": "https://example.com/models/8E32.glb",
}

# ========================
# CV CALCULATION MODULE WITH SEPARATE INLET/OUTLET PIPE SIZES
# ========================
def reynolds_number(flow_m3h: float, d_m: float, visc_cst: float, 
                   F_d: float, F_L: float, C_v: float) -> float:
    """
    Calculate Reynolds number according to IEC standard formula:
    """
    if visc_cst < 0.1:
        return 1e6
    
    N2 = CONSTANTS["N2"]["mm"]  # Use mm constant
    N4 = CONSTANTS["N4"]["mm"]  # Use mm constant

    D_mm = d_m * 1000  # Convert diameter to mm
    term = ((F_L**2 * C_v**2) / (N2 * D_mm**4)) + 1
    denominator = visc_cst * math.sqrt(F_L) * math.sqrt(C_v)
    
    return (N4 * F_d * flow_m3h) / denominator * term

def viscosity_correction(rev: float, method: str = "size_selection") -> float:
    """
    Calculate FR based on Reynolds number and method:
    - 'size_selection': For valve size selection
    - 'flow_prediction': For flow rate prediction
    - 'pressure_prediction': For pressure drop prediction
    """
    if rev >= 40000:
        return 1.0
    
    if method == "size_selection":
        if rev < 56:
            return 0.019 * (rev ** 0.67)
        else:
            # Use table values with linear interpolation
            rev_points = [56, 66, 79, 94, 110, 130, 154, 188, 230, 278, 340, 
                         471, 620, 980, 1560, 2470, 4600, 10200, 40000]
            fr_points = [0.284, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60,
                        0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 1.00]
            return np.interp(rev, rev_points, fr_points)
    
    return 1.0

def calculate_piping_factor_fp_separate(valve_d_inch: float, pipe_d_in_inch: float, 
                                       pipe_d_out_inch: float, cv_op: float, 
                                       record: CalculationRecord = None) -> tuple:
    """
    Calculate piping geometry factor Fp with separate inlet and outlet pipe diameters
    According to IEC 60534-2-1 standard
    
    Returns: (Fp, fp_details) where fp_details is a dictionary with calculation details
    """
    if (pipe_d_in_inch <= valve_d_inch and pipe_d_out_inch <= valve_d_inch) or \
       (abs(pipe_d_in_inch - valve_d_inch) < 0.01 and abs(pipe_d_out_inch - valve_d_inch) < 0.01):
        fp_details = {
            'valve_d_inch': valve_d_inch,
            'pipe_d_in_inch': pipe_d_in_inch,
            'pipe_d_out_inch': pipe_d_out_inch,
            'cv_op': cv_op,
            'd_ratio_in': valve_d_inch / pipe_d_in_inch,
            'd_ratio_out': valve_d_inch / pipe_d_out_inch,
            'K1': 0.0,
            'K2': 0.0,
            'KB1': 0.0,
            'KB2': 0.0,
            'sumK': 0.0,
            'N2': CONSTANTS["N2"]["inch"],
            'term': 1.0,
            'Fp': 1.0,
            'note': 'No reducers/expanders needed (pipe diameter ≤ valve diameter)'
        }
        if record:
            record.add_step(
                step_number=record.iteration_count + 1,
                description="Calculate piping geometry factor Fp",
                formula="Fp = 1.0 (no reducers/expanders needed)",
                result="1.000",
                unit="",
                details=fp_details
            )
        return 1.0, fp_details
    
    # Calculate diameter ratios
    d_ratio_in = valve_d_inch / pipe_d_in_inch
    d_ratio_out = valve_d_inch / pipe_d_out_inch
    
    # Calculate inlet reducer/expander coefficient K1
    K1 = 0.5 * (1 - d_ratio_in**2)**2 if d_ratio_in < 1 else 0
    
    # Calculate outlet reducer/expander coefficient K2
    K2 = 1.0 * (1 - d_ratio_out**2)**2 if d_ratio_out < 1 else 0
    
    # Calculate Bernoulli coefficients
    KB1 = 1 - d_ratio_in**4
    KB2 = 1 - d_ratio_out**4
    
    # Sum of loss coefficients
    sumK = K1 + K2 + KB1 - KB2
    
    # Get N2 constant (using inch units)
    N2 = CONSTANTS["N2"]["inch"]
    
    # Calculate term for Fp formula
    term = 1 + (sumK / N2) * (cv_op / valve_d_inch**2)**2
    
    # Calculate Fp
    Fp = 1 / math.sqrt(term) if term > 0 else 1.0
    
    # Create detailed information
    fp_details = {
        'valve_d_inch': valve_d_inch,
        'pipe_d_in_inch': pipe_d_in_inch,
        'pipe_d_out_inch': pipe_d_out_inch,
        'cv_op': cv_op,
        'd_ratio_in': d_ratio_in,
        'd_ratio_out': d_ratio_out,
        'K1': K1,
        'K2': K2,
        'KB1': KB1,
        'KB2': KB2,
        'sumK': sumK,
        'N2': N2,
        'term': term,
        'Fp': Fp,
        'formula': 'Fp = 1 / √(1 + (ΣK/N2) * (Cv/d²)²)',
        'components': 'ΣK = K1 + K2 + KB1 - KB2',
        'K1_formula': 'K1 = 0.5 * (1 - (d_valve/d_inlet)²)² (if d_valve < d_inlet)',
        'K2_formula': 'K2 = 1.0 * (1 - (d_valve/d_outlet)²)² (if d_valve < d_outlet)',
        'KB1_formula': 'KB1 = 1 - (d_valve/d_inlet)⁴',
        'KB2_formula': 'KB2 = 1 - (d_valve/d_outlet)⁴'
    }
    
    # Add to calculation record if provided
    if record:
        record.add_step(
            step_number=record.iteration_count + 1,
            description="Calculate piping geometry factor Fp",
            formula="Fp = 1 / √(1 + (ΣK/N2) * (Cv/d²)²) where ΣK = K1 + K2 + KB1 - KB2",
            result=f"{Fp:.4f}",
            unit="",
            details=fp_details
        )
        record.set_fp_details(fp_details)
        record.add_formula(
            name="Piping Geometry Factor Fp",
            formula="Fp = 1 / √(1 + (ΣK/N2) * (Cv/d²)²)",
            explanation="Accounts for pressure loss due to pipe reducers/expanders",
            variables={
                "ΣK": "Sum of loss coefficients = K1 + K2 + KB1 - KB2",
                "N2": "Numerical constant (890 for inch units)",
                "Cv": "Valve flow coefficient at operating point",
                "d": "Valve internal diameter (inch)"
            }
        )
    
    return Fp, fp_details

def calculate_flp_separate(valve, valve_d_inch: float, pipe_d_in_inch: float, 
                          pipe_d_out_inch: float, cv_op: float, fl_at_op: float,
                          record: CalculationRecord = None) -> tuple:
    """
    Calculate combined liquid pressure recovery factor FLP with separate inlet/outlet pipes
    
    Returns: (FLP, flp_details) where flp_details is a dictionary with calculation details
    """
    if abs(pipe_d_in_inch - valve_d_inch) < 0.01 and abs(pipe_d_out_inch - valve_d_inch) < 0.01:
        flp_details = {
            'valve_d_inch': valve_d_inch,
            'pipe_d_in_inch': pipe_d_in_inch,
            'pipe_d_out_inch': pipe_d_out_inch,
            'cv_op': cv_op,
            'fl_at_op': fl_at_op,
            'd_ratio_in': valve_d_inch / pipe_d_in_inch,
            'K1': 0.0,
            'KB1': 0.0,
            'Ki': 0.0,
            'N2': CONSTANTS["N2"]["inch"],
            'term': 1/fl_at_op**2,
            'FLP': fl_at_op,
            'note': 'No reducers needed (pipe diameter = valve diameter)'
        }
        if record:
            record.add_step(
                step_number=record.iteration_count + 1,
                description="Calculate combined liquid pressure recovery factor FLP",
                formula="FLP = Fl (no reducers needed)",
                result=f"{fl_at_op:.4f}",
                unit="",
                details=flp_details
            )
        return fl_at_op, flp_details
    
    # Calculate inlet reducer/expander coefficient K1
    d_ratio_in = valve_d_inch / pipe_d_in_inch
    K1 = 0.5 * (1 - d_ratio_in**2)**2 if d_ratio_in < 1 else 0
    
    # Calculate Bernoulli coefficient for inlet
    KB1 = 1 - d_ratio_in**4
    
    # Ki = K1 + KB1 (only inlet contributes to FLP per IEC standard)
    Ki = K1 + KB1
    
    # Get N2 constant (using inch units)
    N2 = CONSTANTS["N2"]["inch"]
    
    # Calculate term for FLP formula
    term = (Ki / N2) * (cv_op / valve_d_inch**2)**2 + 1/fl_at_op**2
    
    # Calculate FLP
    FLP = 1 / math.sqrt(term) if term > 0 else fl_at_op
    
    # Create detailed information
    flp_details = {
        'valve_d_inch': valve_d_inch,
        'pipe_d_in_inch': pipe_d_in_inch,
        'pipe_d_out_inch': pipe_d_out_inch,
        'cv_op': cv_op,
        'fl_at_op': fl_at_op,
        'd_ratio_in': d_ratio_in,
        'K1': K1,
        'KB1': KB1,
        'Ki': Ki,
        'N2': N2,
        'term': term,
        'FLP': FLP,
        'formula': 'FLP = 1 / √((Ki/N2)*(Cv/d²)² + 1/Fl²)',
        'Ki_formula': 'Ki = K1 + KB1',
        'K1_formula': 'K1 = 0.5 * (1 - (d_valve/d_inlet)²)² (if d_valve < d_inlet)',
        'KB1_formula': 'KB1 = 1 - (d_valve/d_inlet)⁴'
    }
    
    # Add to calculation record if provided
    if record:
        record.add_step(
            step_number=record.iteration_count + 1,
            description="Calculate combined liquid pressure recovery factor FLP",
            formula="FLP = 1 / √((Ki/N2)*(Cv/d²)² + 1/Fl²) where Ki = K1 + KB1",
            result=f"{FLP:.4f}",
            unit="",
            details=flp_details
        )
    
    return FLP, flp_details

def calculate_x_tp_separate(valve, valve_d_inch: float, pipe_d_in_inch: float, 
                           pipe_d_out_inch: float, fp: float, cv_op: float, xt_at_op: float,
                           record: CalculationRecord = None) -> tuple:
    """
    Calculate combined pressure drop ratio factor xTP with separate inlet/outlet pipes
    
    Returns: (xTP, xtp_details) where xtp_details is a dictionary with calculation details
    """
    if abs(pipe_d_in_inch - valve_d_inch) < 0.01 and abs(pipe_d_out_inch - valve_d_inch) < 0.01:
        xtp_details = {
            'valve_d_inch': valve_d_inch,
            'pipe_d_in_inch': pipe_d_in_inch,
            'pipe_d_out_inch': pipe_d_out_inch,
            'fp': fp,
            'cv_op': cv_op,
            'xt_at_op': xt_at_op,
            'd_ratio_in': valve_d_inch / pipe_d_in_inch,
            'K1': 0.0,
            'KB1': 0.0,
            'Ki': 0.0,
            'N5': CONSTANTS["N5"]["inch"],
            'term': 1.0,
            'xTP': xt_at_op / fp**2,
            'note': 'No reducers needed (pipe diameter = valve diameter)'
        }
        if record:
            record.add_step(
                step_number=record.iteration_count + 1,
                description="Calculate combined pressure drop ratio factor xTP",
                formula="xTP = xT / Fp² (no reducers needed)",
                result=f"{xt_at_op / fp**2:.4f}",
                unit="",
                details=xtp_details
            )
        return xt_at_op / fp**2, xtp_details
    
    # Calculate inlet reducer/expander coefficient K1
    d_ratio_in = valve_d_inch / pipe_d_in_inch
    K1 = 0.5 * (1 - d_ratio_in**2)**2 if d_ratio_in < 1 else 0
    
    # Calculate Bernoulli coefficient for inlet
    KB1 = 1 - d_ratio_in**4
    
    # Ki = K1 + KB1 (only inlet contributes to xTP per IEC standard)
    Ki = K1 + KB1
    
    # Get N5 constant (using inch units)
    N5 = CONSTANTS["N5"]["inch"]
    
    # Calculate term for xTP formula
    term = 1 + (xt_at_op * Ki / N5) * (cv_op / valve_d_inch**2)**2
    
    # Calculate xTP
    xTP = xt_at_op / fp**2 * (1 / term) if term > 0 else xt_at_op / fp**2
    
    # Create detailed information
    xtp_details = {
        'valve_d_inch': valve_d_inch,
        'pipe_d_in_inch': pipe_d_in_inch,
        'pipe_d_out_inch': pipe_d_out_inch,
        'fp': fp,
        'cv_op': cv_op,
        'xt_at_op': xt_at_op,
        'd_ratio_in': d_ratio_in,
        'K1': K1,
        'KB1': KB1,
        'Ki': Ki,
        'N5': N5,
        'term': term,
        'xTP': xTP,
        'formula': 'xTP = xT / Fp² * (1 / (1 + (xT * Ki / N5) * (Cv/d²)²))',
        'Ki_formula': 'Ki = K1 + KB1',
        'K1_formula': 'K1 = 0.5 * (1 - (d_valve/d_inlet)²)² (if d_valve < d_inlet)',
        'KB1_formula': 'KB1 = 1 - (d_valve/d_inlet)⁴'
    }
    
    # Add to calculation record if provided
    if record:
        record.add_step(
            step_number=record.iteration_count + 1,
            description="Calculate combined pressure drop ratio factor xTP",
            formula="xTP = xT / Fp² * (1 / (1 + (xT * Ki / N5) * (Cv/d²)²)) where Ki = K1 + KB1",
            result=f"{xTP:.4f}",
            unit="",
            details=xtp_details
        )
    
    return xTP, xtp_details

# ========================
# UPDATED VELOCITY CALCULATION FUNCTIONS WITH PRESSURE-DEPENDENT VOLUMETRIC FLOW
# ========================
def calculate_volumetric_flow_at_conditions(scenario, pressure_bar, temperature_c):
    """
    Calculate actual volumetric flow rate (m³/h) at specific pressure and temperature
    Accounts for compressibility for gases and steam
    """
    fluid_type = scenario["fluid_type"]
    
    if fluid_type == "liquid":
        # For liquids, volumetric flow is constant (incompressible)
        return scenario["flow"]  # Already in m³/h
    
    elif fluid_type == "gas":
        # For gases: Q_actual = Q_std * (P_std/P_actual) * (T_actual/T_std) * (Z_actual/Z_std)
        Q_std = scenario["flow"]  # std m³/h
        
        # Standard conditions
        T_std = STANDARD_TEMPERATURE  # K
        P_std = STANDARD_PRESSURE  # bar
        
        # Actual conditions
        T_actual = temperature_c + C_TO_K  # K
        P_actual = pressure_bar  # bar
        
        # Calculate compressibility factors
        fluid_name = scenario.get('fluid_library', 'air')
        if fluid_name in FLUID_LIBRARY:
            Z_actual = calculate_compressibility_factor(fluid_name, temperature_c, pressure_bar)
            # Z at standard conditions (approximate as 1.0 for ideal gas at standard conditions)
            Z_std = 1.0
        else:
            Z_actual = scenario.get('z', 1.0)
            Z_std = 1.0
        
        # Apply gas law correction
        Q_actual = Q_std * (P_std / P_actual) * (T_actual / T_std) * (Z_actual / Z_std)
        return Q_actual
    
    elif fluid_type == "steam":
        # For steam: Q = mass_flow / density
        mass_flow = scenario["flow"]  # kg/h
        
        # Get density at actual conditions
        fluid_name = scenario.get('fluid_library', 'Water')
        if fluid_name in FLUID_LIBRARY:
            density = calculate_density(fluid_name, temperature_c, pressure_bar)
        else:
            density = scenario.get('rho', 1.0)  # kg/m³
        
        if density > 0:
            Q_actual = mass_flow / density  # m³/h
        else:
            Q_actual = 0
        
        return Q_actual
    
    return scenario["flow"]

def calculate_velocity_and_pressure_drop(scenario, pipe_diameter_inch, pressure_bar, 
                                        temperature_c, location="inlet"):
    """
    Calculate velocity and velocity pressure drop at specific pipe location
    """
    # Calculate volumetric flow at these conditions
    Q_m3h = calculate_volumetric_flow_at_conditions(scenario, pressure_bar, temperature_c)
    
    # Convert pipe diameter to meters
    d_m = pipe_diameter_inch * 0.0254
    area = math.pi * (d_m / 2)**2
    
    # Convert flow from m³/h to m³/s
    Q_m3s = Q_m3h / 3600
    
    # Calculate velocity
    velocity = Q_m3s / area if area > 0 else 0
    
    # Get density at these conditions
    fluid_type = scenario["fluid_type"]
    fluid_name = scenario.get('fluid_library')
    
    if fluid_type == "liquid":
        if fluid_name in FLUID_LIBRARY:
            density = calculate_density(fluid_name, temperature_c, pressure_bar)
        else:
            density = scenario.get('sg', 1000) * 1000  # Assume water density
    
    elif fluid_type == "gas":
        if fluid_name in FLUID_LIBRARY:
            density = calculate_density(fluid_name, temperature_c, pressure_bar)
        else:
            # Ideal gas law: ρ = (P * M) / (R * T)
            # where M = sg * 28.97 (molar mass of air in g/mol)
            sg = scenario.get('sg', 1.0)
            M = sg * 28.97 / 1000  # kg/mol
            R = 8.314462618  # J/(mol·K)
            T = temperature_c + C_TO_K  # K
            P = pressure_bar * 1e5  # Pa
            density = (P * M) / (R * T)
    
    elif fluid_type == "steam":
        if fluid_name in FLUID_LIBRARY:
            density = calculate_density(fluid_name, temperature_c, pressure_bar)
        else:
            density = scenario.get('rho', 1.0)
    
    else:
        density = 1000  # Default
    
    # Calculate velocity pressure drop (dynamic pressure): ΔP = 0.5 * ρ * v²
    pressure_drop_pa = 0.5 * density * velocity**2
    pressure_drop_bar = pressure_drop_pa * 1e-5
    
    return velocity, pressure_drop_bar, Q_m3h, density

def calculate_valve_velocity_separate(scenario, valve, op_point):
    """
    Calculate flow velocities at valve opening point with separate pipe sizes
    CORRECTED VERSION: Accounts for different volumetric flows at inlet and outlet
    """
    # Get Cv at operating point
    cv_op = valve.get_cv_at_opening(op_point)
    cv_100 = valve.get_cv_at_opening(100)
    
    # Calculate valve area at full opening (inlet area)
    diameter_in = valve.diameter
    area_full = math.pi * (diameter_in * 0.0254 / 2)**2  # in m²
    
    # Calculate flow area at operating point (orifice area)
    if cv_100 > 0:
        area_op = area_full * (cv_op / cv_100)
    else:
        area_op = area_full
    
    # Get pipe diameters from scenario
    use_valve_size = scenario.get("use_valve_size", True)
    if use_valve_size:
        pipe_d_in = valve.diameter
        pipe_d_out = valve.diameter
    else:
        pipe_d_in = scenario["pipe_d_in"]
        pipe_d_out = scenario["pipe_d_out"]
    
    # Get conditions
    p1 = scenario["p1"]
    p2 = scenario["p2"]
    temp = scenario["temp"]
    
    # INLET calculations (at P1, T1)
    inlet_velocity, inlet_pressure_drop, Q_in_m3h, inlet_density = \
        calculate_velocity_and_pressure_drop(scenario, pipe_d_in, p1, temp, "inlet")
    
    # OUTLET calculations (at P2, T2)
    # For gases/steam, temperature might change due to expansion
    # Use isenthalpic expansion approximation: T2 ≈ T1 for small pressure drops
    outlet_velocity, outlet_pressure_drop, Q_out_m3h, outlet_density = \
        calculate_velocity_and_pressure_drop(scenario, pipe_d_out, p2, temp, "outlet")
    
    # ORIFICE calculations (at valve, assume average conditions)
    # For compressible fluids, use average of inlet and outlet
    p_avg = (p1 + p2) / 2
    Q_orifice_m3h = calculate_volumetric_flow_at_conditions(scenario, p_avg, temp)
    Q_orifice_m3s = Q_orifice_m3h / 3600
    orifice_velocity = Q_orifice_m3s / area_op if area_op > 0 else 0
    
    # Calculate orifice density
    if scenario["fluid_type"] == "liquid":
        orifice_density = inlet_density  # Constant for liquids
    else:
        orifice_density = (inlet_density + outlet_density) / 2
    
    orifice_pressure_drop = 0.5 * orifice_density * orifice_velocity**2 * 1e-5
    
    # Check against limits
    warnings = []
    if inlet_velocity > VELOCITY_LIMITS.get(scenario["fluid_type"], 10):
        warnings.append(f"High inlet velocity ({inlet_velocity:.1f} m/s) for {scenario['fluid_type']}! (max {VELOCITY_LIMITS.get(scenario['fluid_type'], 10)} m/s)")
    if outlet_velocity > VELOCITY_LIMITS.get(scenario["fluid_type"], 10):
        warnings.append(f"High outlet velocity ({outlet_velocity:.1f} m/s) for {scenario['fluid_type']}! (max {VELOCITY_LIMITS.get(scenario['fluid_type'], 10)} m/s)")
    if orifice_velocity > VELOCITY_LIMITS.get(scenario["fluid_type"], 10) * 2:
        warnings.append(f"High orifice velocity ({orifice_velocity:.1f} m/s) for {scenario['fluid_type']}!")
    
    velocity_warning = "; ".join(warnings) if warnings else ""
    
    return {
        'orifice_velocity': orifice_velocity,
        'inlet_velocity': inlet_velocity,
        'outlet_velocity': outlet_velocity,
        'inlet_pressure_drop': inlet_pressure_drop,
        'outlet_pressure_drop': outlet_pressure_drop,
        'orifice_pressure_drop': orifice_pressure_drop,
        'warning': velocity_warning,
        'inlet_density': inlet_density,
        'outlet_density': outlet_density,
        'orifice_density': orifice_density,
        'inlet_flow_m3h': Q_in_m3h,
        'outlet_flow_m3h': Q_out_m3h,
        'orifice_flow_m3h': Q_orifice_m3h,
        'pressure_ratio': p2 / p1 if p1 > 0 else 0,
        'density_ratio': outlet_density / inlet_density if inlet_density > 0 else 0,
        'flow_ratio': Q_out_m3h / Q_in_m3h if Q_in_m3h > 0 else 0
    }

# ========================
# CV_LIQUID FUNCTION WITH ENHANCED CALCULATION RECORD
# ========================
def cv_liquid_with_record(flow: float, p1: float, p2: float, sg: float, fl_at_op: float, 
                         pv: float, pc: float, visc_cst: float, d_m: float, 
                         valve, fp: float = 1.0, cv_op: float = None,
                         record: CalculationRecord = None) -> tuple:
    """
    Calculate required Cv for liquid with enhanced calculation record
    """
    # Store input parameters
    if record:
        record.set_input_parameter("Flow rate", flow, "m³/h")
        record.set_input_parameter("Inlet pressure P1", p1, "bar")
        record.set_input_parameter("Outlet pressure P2", p2, "bar")
        record.set_input_parameter("Specific gravity SG", sg, "")
        record.set_input_parameter("Fl at operating point", fl_at_op, "")
        record.set_input_parameter("Vapor pressure Pv", pv, "bar")
        record.set_input_parameter("Critical pressure Pc", pc, "bar")
        record.set_input_parameter("Kinematic viscosity", visc_cst, "cSt")
        record.set_input_parameter("Valve diameter", d_m * 1000, "mm")
        record.set_input_parameter("Piping factor Fp", fp, "")
        
        record.add_formula(
            "Liquid Cv Calculation",
            "Cv = (Q / N1) * √(SG / ΔP)",
            "Basic Cv formula for liquids (ISA/IEC standard)",
            {"Q": "Flow rate", "N1": "Numerical constant", "SG": "Specific gravity", "ΔP": "Pressure drop"}
        )
        record.add_formula(
            "Choked Flow Check",
            "ΔP_max = Fl² * (P1 - FF * Pv)",
            "Maximum pressure drop before choked flow occurs",
            {"Fl": "Liquid pressure recovery factor", "FF": "Critical pressure ratio", "Pv": "Vapor pressure"}
        )
        record.add_formula(
            "Critical Pressure Ratio FF",
            "FF = 0.96 - 0.28 * √(Pv / Pc)",
            "Ratio of vapor pressure to critical pressure",
            {"Pv": "Vapor pressure", "Pc": "Critical pressure"}
        )
        record.add_formula(
            "Reynolds Number for Liquids",
            "Re = (N4 * Fd * Q) / (ν * √Fl * √Cv) * ((Fl² * Cv²)/(N2 * D⁴) + 1)",
            "Calculates Reynolds number for viscosity correction",
            {"N4": "Numerical constant", "Fd": "Valve style modifier", "ν": "Kinematic viscosity"}
        )
    
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        if record:
            record.add_warning("Invalid pressure values: P1 <= P2 or negative", "error", 
                              {"P1": p1, "P2": p2})
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 
                  'reynolds': 0, 'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    N1 = CONSTANTS["N1"]["m³/h, bar"]
    dp = p1 - p2
    
    if record:
        record.add_step(1, "Calculate pressure drop", "ΔP = P1 - P2", 
                       f"{dp:.3f}", "bar", {"P1": p1, "P2": p2, "ΔP": dp})
    
    if dp <= 0:
        if record:
            record.add_warning("No pressure drop (ΔP <= 0)", "error", {"ΔP": dp})
        return 0, {'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 'reynolds': 0, 
                  'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    # Calculate Ff (fluid property only)
    ff = calculate_ff(pv, pc)
    if record:
        record.add_step(2, "Calculate critical pressure ratio FF", 
                       "FF = 0.96 - 0.28 * √(Pv / Pc)", 
                       f"{ff:.4f}", "", 
                       {"Pv": pv, "Pc": pc, "FF": ff})
        record.add_assumption(f"FF = {ff:.4f}", "Based on vapor and critical pressures", 
                            {"Pv": pv, "Pc": pc, "FF": ff})
    
    # Calculate theoretical Cv (valve-independent)
    dp_max_fluid = p1 - ff * pv
    if dp_max_fluid <= 0:
        dp_max_fluid = float('inf')
    
    if dp < dp_max_fluid:
        theoretical_cv = (flow / N1) * math.sqrt(sg / dp)
        if record:
            record.add_step(3, "Calculate theoretical Cv (unchoked)", 
                           "Cv = (Q / N1) * √(SG / ΔP)", 
                           f"{theoretical_cv:.2f}", "",
                           {"Q": flow, "N1": N1, "SG": sg, "ΔP": dp, "Cv": theoretical_cv})
    else:
        theoretical_cv = (flow / N1) * math.sqrt(sg) / math.sqrt(dp_max_fluid)
        if record:
            record.add_step(3, "Calculate theoretical Cv (choked)", 
                           "Cv = (Q / N1) * √(SG) / √(ΔP_max_fluid)", 
                           f"{theoretical_cv:.2f}", "",
                           {"Q": flow, "N1": N1, "SG": sg, "ΔP_max_fluid": dp_max_fluid, "Cv": theoretical_cv})
            record.add_warning("Flow is choked at fluid level", "warning",
                             {"ΔP": dp, "ΔP_max_fluid": dp_max_fluid})
    
    # Calculate max allowable pressure drop for the valve (uses Fl)
    dp_max_valve = fl_at_op**2 * (p1 - ff * pv)
    if dp_max_valve <= 0:
        dp_max_valve = float('inf')
    
    if record:
        record.add_step(4, "Calculate maximum valve pressure drop", 
                       "ΔP_max_valve = Fl² * (P1 - FF * Pv)", 
                       f"{dp_max_valve:.3f}", "bar",
                       {"Fl": fl_at_op, "P1": p1, "FF": ff, "Pv": pv, "ΔP_max_valve": dp_max_valve})
    
    # Calculate pseudo Cv (value-specific but without viscosity correction)
    if dp < dp_max_valve:
        cv_pseudo = (flow / N1) * math.sqrt(sg / dp)
        if record:
            record.add_step(5, "Calculate pseudo Cv (unchoked)", 
                           "Cv_pseudo = (Q / N1) * √(SG / ΔP)", 
                           f"{cv_pseudo:.2f}", "",
                           {"Q": flow, "N1": N1, "SG": sg, "ΔP": dp, "Cv_pseudo": cv_pseudo})
    else:
        cv_pseudo = (flow / N1) * math.sqrt(sg) / (fl_at_op * math.sqrt(p1 - ff * pv))
        if record:
            record.add_step(5, "Calculate pseudo Cv (choked)", 
                           "Cv_pseudo = (Q / N1) * √(SG) / (Fl * √(P1 - FF * Pv))", 
                           f"{cv_pseudo:.2f}", "",
                           {"Q": flow, "N1": N1, "SG": sg, "Fl": fl_at_op, "P1": p1, 
                            "FF": ff, "Pv": pv, "Cv_pseudo": cv_pseudo})
            record.add_warning("Flow is choked at valve level", "warning",
                             {"ΔP": dp, "ΔP_max_valve": dp_max_valve})
    
    # Calculate Reynolds number
    rev = reynolds_number(
        flow_m3h=flow,
        d_m=d_m,
        visc_cst=visc_cst,
        F_d=valve.fd,
        F_L=fl_at_op,
        C_v=cv_pseudo
    )
    
    if record:
        N2 = CONSTANTS["N2"]["mm"]
        N4 = CONSTANTS["N4"]["mm"]
        D_mm = d_m * 1000
        term = ((fl_at_op**2 * cv_pseudo**2) / (N2 * D_mm**4)) + 1
        denominator = visc_cst * math.sqrt(fl_at_op) * math.sqrt(cv_pseudo)
        reynolds_detail = f"Re = ({N4} * {valve.fd} * {flow}) / ({visc_cst} * √{fl_at_op} * √{cv_pseudo}) * (({fl_at_op}² * {cv_pseudo}²)/({N2} * {D_mm}⁴) + 1)"
        
        record.add_step(6, "Calculate Reynolds number", 
                       reynolds_detail, 
                       f"{rev:.0f}", "",
                       {"N4": N4, "Fd": valve.fd, "Q": flow, "ν": visc_cst, "Fl": fl_at_op,
                        "Cv": cv_pseudo, "N2": N2, "D": D_mm, "Re": rev})
        record.add_assumption(f"Re = {rev:.0f}", "For viscosity correction calculation",
                            {"Re": rev})
        record.set_intermediate_value("Reynolds Number", rev, "", "For viscosity correction")
    
    # Use viscosity correction for valve size selection
    fr = viscosity_correction(rev, method="size_selection")
    
    if record:
        record.add_step(7, "Calculate viscosity correction factor Fr", 
                       "Fr = f(Re) from IEC table", 
                       f"{fr:.4f}", "",
                       {"Re": rev, "Fr": fr})
        if rev < 40000:
            record.add_warning(f"Low Reynolds number ({rev:.0f}), applying viscosity correction", 
                             "info", {"Re": rev})
        record.set_intermediate_value("Viscosity Correction Fr", fr, "", "Correction factor for low Reynolds number")
    
    # Apply corrections
    cv_after_fp = cv_pseudo / fp
    corrected_cv = cv_after_fp / fr
    
    if record:
        record.add_step(8, "Apply piping correction Fp", 
                       "Cv_after_Fp = Cv_pseudo / Fp", 
                       f"{cv_after_fp:.2f}", "",
                       {"Cv_pseudo": cv_pseudo, "Fp": fp, "Cv_after_Fp": cv_after_fp})
        record.add_step(9, "Apply viscosity correction Fr", 
                       "Cv_corrected = Cv_after_Fp / Fr", 
                       f"{corrected_cv:.2f}", "",
                       {"Cv_after_Fp": cv_after_fp, "Fr": fr, "Cv_corrected": corrected_cv})
        
        # Store all results
        record.set_result("Required Cv", corrected_cv, "", "Final Cv value after all corrections")
        record.set_result("Theoretical Cv", theoretical_cv, "", "Theoretical Cv without corrections")
        record.set_result("Pseudo Cv", cv_pseudo, "", "Cv before viscosity correction")
        record.set_result("Cv after Fp", cv_after_fp, "", "Cv after piping correction")
        record.set_result("Reynolds Number", rev, "", "Dimensionless flow parameter")
        record.set_result("Viscosity Correction Fr", fr, "", "Correction factor for viscosity")
        record.set_result("Critical Pressure Ratio FF", ff, "", "Ratio of vapor to critical pressure")
        record.set_result("Maximum Pressure Drop", dp_max_valve, "bar", "Max ΔP before choked flow")
        
        # Store intermediate values
        record.set_intermediate_value("Pressure Drop ΔP", dp, "bar", "Actual pressure drop")
        record.set_intermediate_value("ΔP_max_fluid", dp_max_fluid, "bar", "Fluid choked flow limit")
        record.set_intermediate_value("Cv_pseudo", cv_pseudo, "", "Intermediate Cv value")
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'fr': fr,
        'reynolds': rev,
        'is_choked': (dp >= dp_max_valve),
        'ff': ff,
        'dp_max': dp_max_valve,
        'fl_at_op': fl_at_op,
        'cv_used_for_fp': cv_op,
        'calculation_steps': record.get_summary() if record else None,
        'cv_pseudo': cv_pseudo,
        'cv_after_fp': cv_after_fp,
        'dp': dp,
        'dp_max_fluid': dp_max_fluid
    }
    
    return corrected_cv, details

# ========================
# CV_GAS FUNCTION WITH ENHANCED CALCULATION RECORD
# ========================
def cv_gas_with_record(flow: float, p1: float, p2: float, sg: float, t: float, k: float, 
                      xt_at_op: float, z: float, fp: float = 1.0, op_point: float = None,
                      cv_op: float = None, record: CalculationRecord = None) -> tuple:
    """
    Calculate required Cv for gas with enhanced calculation record
    """
    # Store input parameters
    if record:
        record.set_input_parameter("Flow rate", flow, "std m³/h")
        record.set_input_parameter("Inlet pressure P1", p1, "bar")
        record.set_input_parameter("Outlet pressure P2", p2, "bar")
        record.set_input_parameter("Specific gravity SG", sg, "")
        record.set_input_parameter("Temperature", t, "°C")
        record.set_input_parameter("Specific heat ratio k", k, "")
        record.set_input_parameter("Pressure drop ratio factor xT", xt_at_op, "")
        record.set_input_parameter("Compressibility factor Z", z, "")
        record.set_input_parameter("Piping factor Fp", fp, "")
        
        record.add_formula(
            "Gas Cv Calculation",
            "Cv = (Q / (N7 * Fp * P1 * Y)) * √((SG * T * Z) / X)",
            "Basic Cv formula for gases (ISA/IEC standard)",
            {"Q": "Flow rate", "N7": "Numerical constant", "Fp": "Piping factor", 
             "P1": "Inlet pressure", "Y": "Expansion factor", "SG": "Specific gravity",
             "T": "Temperature (K)", "Z": "Compressibility factor", "X": "Pressure drop ratio"}
        )
        record.add_formula(
            "Expansion Factor Y",
            "Y = 1 - X / (3 * Fk * xT)",
            "Factor accounting for gas expansion",
            {"X": "Pressure drop ratio", "Fk": "Ratio of specific heats", "xT": "Pressure drop ratio factor"}
        )
        record.add_formula(
            "Critical Pressure Ratio",
            "X_crit = Fk * xT",
            "Critical pressure drop ratio for choked flow",
            {"Fk": "Ratio of specific heats", "xT": "Pressure drop ratio factor"}
        )
        record.add_formula(
            "Ratio of Specific Heats Fk",
            "Fk = k / 1.4",
            "Ratio of fluid specific heat to air specific heat",
            {"k": "Specific heat ratio"}
        )
    
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        if record:
            record.add_warning("Invalid pressure values: P1 <= P2 or negative", "error",
                              {"P1": p1, "P2": p2})
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': 0, 
                  'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    x_actual = (p1 - p2) / p1
    if record:
        record.add_step(1, "Calculate pressure drop ratio", 
                       "X = (P1 - P2) / P1", 
                       f"{x_actual:.4f}", "",
                       {"P1": p1, "P2": p2, "X": x_actual})
        record.set_intermediate_value("Pressure Drop Ratio X", x_actual, "", "Actual X = ΔP/P1")
    
    if x_actual <= 0:
        if record:
            record.add_warning("Negative pressure drop ratio", "error",
                              {"X": x_actual})
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': x_actual, 
                  'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    fk = k / 1.4
    x_crit = fk * xt_at_op
    
    if record:
        record.add_step(2, "Calculate k ratio", 
                       "Fk = k / 1.4", 
                       f"{fk:.4f}", "",
                       {"k": k, "1.4": 1.4, "Fk": fk})
        record.add_step(3, "Calculate critical pressure drop ratio", 
                       "X_crit = Fk * xT", 
                       f"{x_crit:.4f}", "",
                       {"Fk": fk, "xT": xt_at_op, "X_crit": x_crit})
        record.add_assumption(f"Fk = {fk:.4f}", "Ratio of specific heat to air (1.4)",
                            {"k": k, "Fk": fk})
        record.set_intermediate_value("Fk", fk, "", "Ratio of specific heats")
        record.set_intermediate_value("Critical X", x_crit, "", "Choked flow limit")
    
    # Standard calculation
    if x_actual >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
        if record:
            record.add_step(4, "Flow is choked, use X_crit", 
                           "X_actual ≥ X_crit", 
                           f"Using X = {x:.4f}, Y = {y:.3f}", "",
                           {"X_actual": x_actual, "X_crit": x_crit, "X": x, "Y": y})
            record.add_warning("Choked flow condition detected", "warning",
                             {"X_actual": x_actual, "X_crit": x_crit})
    else:
        x = x_actual
        y = 1 - x / (3 * fk * xt_at_op)
        is_choked = False
        if record:
            record.add_step(4, "Calculate expansion factor Y", 
                           "Y = 1 - X / (3 * Fk * xT)", 
                           f"{y:.4f}", "",
                           {"X": x, "Fk": fk, "xT": xt_at_op, "Y": y})
    
    N7 = CONSTANTS["N7"]["m³/h, bar, K (standard)"]
    term = (sg * (t + C_TO_K) * z) / x
    
    if record:
        record.add_step(5, "Calculate term inside square root", 
                       "Term = (SG * T * Z) / X", 
                       f"{term:.4f}", "",
                       {"SG": sg, "T": t + C_TO_K, "Z": z, "X": x, "Term": term})
        record.set_intermediate_value("Term inside sqrt", term, "", "Numerator for Cv calculation")
    
    if term < 0:
        if record:
            record.add_warning("Negative value in square root", "error",
                             {"Term": term})
        return 0, {'error': 'Negative value in sqrt', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit, 
                  'x_actual': x_actual, 'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    theoretical_cv = (flow / (N7 * fp * p1 * y)) * math.sqrt(term)
    corrected_cv = theoretical_cv
    
    if record:
        record.add_step(6, "Calculate theoretical Cv", 
                       "Cv = (Q / (N7 * Fp * P1 * Y)) * √(Term)", 
                       f"{theoretical_cv:.2f}", "",
                       {"Q": flow, "N7": N7, "Fp": fp, "P1": p1, "Y": y, 
                        "√Term": math.sqrt(term), "Cv": theoretical_cv})
        
        # Store results
        record.set_result("Required Cv", corrected_cv, "", "Final Cv value")
        record.set_result("Theoretical Cv", theoretical_cv, "", "Theoretical Cv without corrections")
        record.set_result("Expansion Factor Y", y, "", "Gas expansion correction factor")
        record.set_result("Pressure Drop Ratio X", x_actual, "", "Actual X = ΔP/P1")
        record.set_result("Critical X", x_crit, "", "Choked flow limit")
        record.set_result("Is Choked", is_choked, "", "Choked flow status")
        record.set_result("Fk", fk, "", "Ratio of specific heats")
        
        # Store intermediate values
        record.set_intermediate_value("Temperature K", t + C_TO_K, "K", "Absolute temperature")
        record.set_intermediate_value("N7 constant", N7, "", "Numerical constant for gases")
    
    # Alternative calculation using x_actual and constant Y=0.667
    if x_actual > 0:
        y_alt = 0.667
        x_alt = x_actual
        term_alt = (sg * (t + C_TO_K) * z) / x_alt
        if term_alt > 0:
            cv_alternative = (flow / (N7 * fp * p1 * y_alt)) * math.sqrt(term_alt)
            if record:
                record.add_step(7, "Alternative Cv (conservative)", 
                               "Cv_alt = (Q / (N7 * Fp * P1 * 0.667)) * √((SG * T * Z) / X_actual)", 
                               f"{cv_alternative:.2f}", "",
                               {"Q": flow, "N7": N7, "Fp": fp, "P1": p1, "Y_alt": y_alt,
                                "X_actual": x_actual, "√Term_alt": math.sqrt(term_alt), "Cv_alt": cv_alternative})
                record.set_result("Alternative Cv (Y=0.667)", cv_alternative, "", 
                                "Conservative Cv using constant Y=0.667")
        else:
            cv_alternative = 0
    else:
        cv_alternative = 0
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'x_actual': x_actual,
        'xt_at_op': xt_at_op,
        'xt_op_point': op_point,
        'cv_alternative': cv_alternative,
        'alternative_method': f'Using x_actual={x_actual:.4f} and constant Y=0.667',
        'cv_used_for_fp': cv_op,
        'calculation_steps': record.get_summary() if record else None,
        'fk': fk,
        'term_inside_sqrt': term,
        'temperature_k': t + C_TO_K
    }
    
    return corrected_cv, details

# ========================
# CV_STEAM FUNCTION WITH ENHANCED CALCULATION RECORD
# ========================
def cv_steam_with_record(flow: float, p1: float, p2: float, rho: float, k: float, 
                        xt_at_op: float, fp: float = 1.0, op_point: float = None,
                        cv_op: float = None, record: CalculationRecord = None) -> tuple:
    """
    Calculate required Cv for steam with enhanced calculation record
    """
    # Store input parameters
    if record:
        record.set_input_parameter("Mass flow rate", flow, "kg/h")
        record.set_input_parameter("Inlet pressure P1", p1, "bar")
        record.set_input_parameter("Outlet pressure P2", p2, "bar")
        record.set_input_parameter("Density ρ", rho, "kg/m³")
        record.set_input_parameter("Specific heat ratio k", k, "")
        record.set_input_parameter("Pressure drop ratio factor xT", xt_at_op, "")
        record.set_input_parameter("Piping factor Fp", fp, "")
        
        record.add_formula(
            "Steam Cv Calculation",
            "Cv = Q / (N6 * Y * √(X * P1 * ρ))",
            "Basic Cv formula for steam (ISA/IEC standard)",
            {"Q": "Mass flow rate", "N6": "Numerical constant", "Y": "Expansion factor",
             "X": "Pressure drop ratio", "P1": "Inlet pressure", "ρ": "Density"}
        )
        record.add_formula(
            "Expansion Factor Y",
            "Y = 1 - X / (3 * Fk * xT)",
            "Factor accounting for steam expansion",
            {"X": "Pressure drop ratio", "Fk": "Ratio of specific heats", "xT": "Pressure drop ratio factor"}
        )
        record.add_formula(
            "Critical Pressure Ratio",
            "X_crit = Fk * xT",
            "Critical pressure drop ratio for choked flow",
            {"Fk": "Ratio of specific heats", "xT": "Pressure drop ratio factor"}
        )
        record.add_formula(
            "Ratio of Specific Heats Fk",
            "Fk = k / 1.4",
            "Ratio of steam specific heat to air specific heat",
            {"k": "Specific heat ratio"}
        )
    
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        if record:
            record.add_warning("Invalid pressure values: P1 <= P2 or negative", "error",
                              {"P1": p1, "P2": p2})
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': 0, 
                  'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    x_actual = (p1 - p2) / p1
    if record:
        record.add_step(1, "Calculate pressure drop ratio", 
                       "X = (P1 - P2) / P1", 
                       f"{x_actual:.4f}", "",
                       {"P1": p1, "P2": p2, "X": x_actual})
        record.set_intermediate_value("Pressure Drop Ratio X", x_actual, "", "Actual X = ΔP/P1")
    
    if x_actual <= 0:
        if record:
            record.add_warning("Negative pressure drop ratio", "error",
                              {"X": x_actual})
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': x_actual, 
                  'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    fk = k / 1.4
    x_crit = fk * xt_at_op
    
    if record:
        record.add_step(2, "Calculate k ratio", 
                       "Fk = k / 1.4", 
                       f"{fk:.4f}", "",
                       {"k": k, "1.4": 1.4, "Fk": fk})
        record.add_step(3, "Calculate critical pressure drop ratio", 
                       "X_crit = Fk * xT", 
                       f"{x_crit:.4f}", "",
                       {"Fk": fk, "xT": xt_at_op, "X_crit": x_crit})
        record.add_assumption(f"Fk = {fk:.4f}", "Ratio of specific heat to air (1.4)",
                            {"k": k, "Fk": fk})
        record.set_intermediate_value("Fk", fk, "", "Ratio of specific heats")
        record.set_intermediate_value("Critical X", x_crit, "", "Choked flow limit")
    
    # Standard calculation
    if x_actual >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
        if record:
            record.add_step(4, "Flow is choked, use X_crit", 
                           "X_actual ≥ X_crit", 
                           f"Using X = {x:.4f}, Y = {y:.3f}", "",
                           {"X_actual": x_actual, "X_crit": x_crit, "X": x, "Y": y})
            record.add_warning("Choked flow condition detected", "warning",
                             {"X_actual": x_actual, "X_crit": x_crit})
    else:
        x = x_actual
        y = 1 - x / (3 * fk * xt_at_op)
        is_choked = False
        if record:
            record.add_step(4, "Calculate expansion factor Y", 
                           "Y = 1 - X / (3 * Fk * xT)", 
                           f"{y:.4f}", "",
                           {"X": x, "Fk": fk, "xT": xt_at_op, "Y": y})
    
    N6 = CONSTANTS["N6"]["kg/h, bar, kg/m³"]
    term = x * p1 * rho
    
    if record:
        record.add_step(5, "Calculate term inside square root", 
                       "Term = X * P1 * ρ", 
                       f"{term:.4f}", "",
                       {"X": x, "P1": p1, "ρ": rho, "Term": term})
        record.set_intermediate_value("Term inside sqrt", term, "", "Product for Cv calculation")
    
    if term <= 0:
        if record:
            record.add_warning("Invalid term in square root", "error",
                             {"Term": term})
        return 0, {'error': 'Invalid term in sqrt', 'theoretical_cv': 0, 'fp': fp, 
                  'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit, 
                  'x_actual': x_actual, 'xt_at_op': xt_at_op, 'xt_op_point': op_point}
    
    theoretical_cv = flow / (N6 * y * math.sqrt(term))
    corrected_cv = theoretical_cv / fp
    
    if record:
        record.add_step(6, "Calculate theoretical Cv", 
                       "Cv_theo = Q / (N6 * Y * √(Term))", 
                       f"{theoretical_cv:.2f}", "",
                       {"Q": flow, "N6": N6, "Y": y, "√Term": math.sqrt(term), "Cv_theo": theoretical_cv})
        record.add_step(7, "Apply piping correction Fp", 
                       "Cv_corrected = Cv_theo / Fp", 
                       f"{corrected_cv:.2f}", "",
                       {"Cv_theo": theoretical_cv, "Fp": fp, "Cv_corrected": corrected_cv})
        
        # Store results
        record.set_result("Required Cv", corrected_cv, "", "Final Cv value after piping correction")
        record.set_result("Theoretical Cv", theoretical_cv, "", "Theoretical Cv without piping correction")
        record.set_result("Expansion Factor Y", y, "", "Steam expansion correction factor")
        record.set_result("Pressure Drop Ratio X", x_actual, "", "Actual X = ΔP/P1")
        record.set_result("Critical X", x_crit, "", "Choked flow limit")
        record.set_result("Is Choked", is_choked, "", "Choked flow status")
        record.set_result("Fk", fk, "", "Ratio of specific heats")
        
        # Store intermediate values
        record.set_intermediate_value("N6 constant", N6, "", "Numerical constant for steam")
        record.set_intermediate_value("√Term", math.sqrt(term), "", "Square root of product term")
    
    # Alternative calculation using x_actual and constant Y=0.667
    if x_actual > 0:
        y_alt = 0.667
        x_alt = x_actual
        term_alt = x_alt * p1 * rho
        if term_alt > 0:
            cv_alternative = flow / (N6 * y_alt * math.sqrt(term_alt))
            cv_alternative = cv_alternative / fp
            if record:
                record.add_step(8, "Alternative Cv (conservative)", 
                               "Cv_alt = Q / (N6 * 0.667 * √(X_actual * P1 * ρ)) / Fp", 
                               f"{cv_alternative:.2f}", "",
                               {"Q": flow, "N6": N6, "Y_alt": y_alt, "X_actual": x_actual,
                                "P1": p1, "ρ": rho, "Fp": fp, "Cv_alt": cv_alternative})
                record.set_result("Alternative Cv (Y=0.667)", cv_alternative, "", 
                                "Conservative Cv using constant Y=0.667")
        else:
            cv_alternative = 0
    else:
        cv_alternative = 0
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'x_actual': x_actual,
        'xt_at_op': xt_at_op,
        'xt_op_point': op_point,
        'cv_alternative': cv_alternative,
        'alternative_method': f'Using x_actual={x_actual:.4f} and constant Y=0.667',
        'cv_used_for_fp': cv_op,
        'calculation_steps': record.get_summary() if record else None,
        'fk': fk,
        'term_inside_sqrt': term,
        'N6': N6
    }
    
    return corrected_cv, details

def check_cavitation_with_record(p1: float, p2: float, pv: float, fl_at_op: float, pc: float, 
                                record: CalculationRecord = None) -> tuple:
    """
    Check cavitation with enhanced calculation record
    """
    if record:
        record.set_input_parameter("Inlet pressure P1", p1, "bar")
        record.set_input_parameter("Outlet pressure P2", p2, "bar")
        record.set_input_parameter("Vapor pressure Pv", pv, "bar")
        record.set_input_parameter("Liquid recovery factor Fl", fl_at_op, "")
        record.set_input_parameter("Critical pressure Pc", pc, "bar")
        
        record.add_formula(
            "Cavitation Index Sigma (σ)",
            "σ = (P1 - Pv) / ΔP",
            "Ratio of available pressure to pressure drop",
            {"P1": "Inlet pressure", "Pv": "Vapor pressure", "ΔP": "Pressure drop"}
        )
        record.add_formula(
            "Valve Recovery Coefficient Km",
            "Km = Fl²",
            "Square of liquid pressure recovery factor",
            {"Fl": "Liquid pressure recovery factor"}
        )
        record.add_formula(
            "Choked Flow Pressure Drop",
            "ΔP_max = Fl² * (P1 - FF * Pv)",
            "Maximum pressure drop before choked flow",
            {"Fl": "Liquid recovery factor", "FF": "Critical pressure ratio", "Pv": "Vapor pressure"}
        )
    
    if pc <= 0:
        if record:
            record.add_warning("Critical pressure not available", "warning",
                             {"Pc": pc})
        return False, 0, 0, "Critical pressure not available"
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        if record:
            record.add_warning("Invalid pressures", "warning",
                             {"P1": p1, "P2": p2})
        return False, 0, 0, "Invalid pressures"
    
    ff = calculate_ff(pv, pc)
    dp = p1 - p2
    
    if record:
        record.add_step(1, "Calculate critical pressure ratio FF", 
                       "FF = 0.96 - 0.28 * √(Pv / Pc)", 
                       f"{ff:.4f}", "",
                       {"Pv": pv, "Pc": pc, "FF": ff})
        record.add_step(2, "Calculate pressure drop ΔP", 
                       "ΔP = P1 - P2", 
                       f"{dp:.3f}", "bar",
                       {"P1": p1, "P2": p2, "ΔP": dp})
    
    if dp <= 0:
        if record:
            record.add_warning("No pressure drop", "warning",
                             {"ΔP": dp})
        return False, 0, 0, "No pressure drop"
    
    dp_max = fl_at_op**2 * (p1 - ff * pv)
    km = fl_at_op**2
    sigma = (p1 - pv) / dp
    
    if record:
        record.add_step(3, "Calculate sigma (cavitation index)", 
                       "σ = (P1 - Pv) / ΔP", 
                       f"{sigma:.2f}", "",
                       {"P1": p1, "Pv": pv, "ΔP": dp, "σ": sigma})
        record.add_step(4, "Calculate valve recovery coefficient", 
                       "Km = Fl²", 
                       f"{km:.2f}", "",
                       {"Fl": fl_at_op, "Km": km})
        record.add_step(5, "Calculate choked flow pressure drop", 
                       "ΔP_max = Fl² * (P1 - FF * Pv)", 
                       f"{dp_max:.3f}", "bar",
                       {"Fl": fl_at_op, "P1": p1, "FF": ff, "Pv": pv, "ΔP_max": dp_max})
        
        # Store intermediate values
        record.set_intermediate_value("Sigma (σ)", sigma, "", "Cavitation index")
        record.set_intermediate_value("Km", km, "", "Valve recovery coefficient")
        record.set_intermediate_value("ΔP_max", dp_max, "bar", "Maximum pressure drop before choked flow")
        record.set_intermediate_value("FF", ff, "", "Critical pressure ratio")
    
    if dp >= dp_max:
        if record:
            record.add_warning("Choked flow - cavitation likely", "error",
                             {"ΔP": dp, "ΔP_max": dp_max})
        return True, sigma, km, "Choked flow - cavitation likely"
    elif sigma < 1.5 * km:
        if record:
            record.add_warning("Severe cavitation risk", "error",
                             {"σ": sigma, "Km": km, "1.5*Km": 1.5 * km})
        return True, sigma, km, "Severe cavitation risk"
    elif sigma < 2 * km:
        if record:
            record.add_warning("Moderate cavitation risk", "warning",
                             {"σ": sigma, "Km": km, "2*Km": 2 * km})
        return False, sigma, km, "Moderate cavitation risk"
    elif sigma < 4 * km:
        if record:
            record.add_warning("Mild cavitation risk", "info",
                             {"σ": sigma, "Km": km, "4*Km": 4 * km})
        return False, sigma, km, "Mild cavitation risk"
    
    if record:
        record.add_warning("Minimal cavitation risk", "info",
                         {"σ": sigma, "Km": km, "4*Km": 4 * km})
    return False, sigma, km, "Minimal cavitation risk"

# ========================
# ENHANCED PDF REPORT GENERATION
# ========================
class EnhancedPDFReport(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.set_title("Valve Sizing Report with Calculation Details")
        self.set_author("VASTAS Valve Sizing")
        self.alias_nb_pages()
    
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Valve Sizing Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def safe_text(self, text):
        """Convert text to ASCII-safe string"""
        if text is None:
            return ""
        # Remove or replace non-ASCII characters
        safe_str = str(text).encode('ascii', 'ignore').decode('ascii')
        # Replace common special characters with spaces
        safe_str = safe_str.replace('°', ' ').replace('"', ' ').replace("'", " ")
        return safe_str[:500]  # Limit length
    
    def add_section_title(self, title):
        self.ln(5)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.safe_text(title), 0, 1)
        self.ln(2)
    
    def add_subtitle(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, self.safe_text(title), 0, 1)
        self.ln(1)
    
    def add_paragraph(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, self.safe_text(text))
        self.ln(2)
    
    def add_key_value_table(self, data_dict, col_widths=[60, 120]):
        """Add key-value pairs as a table"""
        self.set_font('Arial', '', 9)
        for key, data in data_dict.items():
            value_str = data.get('value', '')
            unit_str = data.get('unit', '')
            desc_str = data.get('description', '')
            
            # Key column
            self.set_font('Arial', 'B', 9)
            self.cell(col_widths[0], 6, self.safe_text(key), 0, 0)
            
            # Value column
            self.set_font('Arial', '', 9)
            value_display = f"{value_str}"
            if unit_str:
                value_display += f" {unit_str}"
            if desc_str:
                value_display += f" ({desc_str})"
            
            self.cell(col_widths[1], 6, self.safe_text(value_display), 0, 1)
        self.ln(3)
    
    def add_calculation_step(self, step_num, description, formula, result, unit="", details=None):
        """Add a calculation step with details"""
        self.set_font('Arial', 'B', 9)
        self.cell(0, 6, f"Step {step_num}: {description}", 0, 1)
        
        self.set_font('Courier', '', 8)
        self.multi_cell(0, 4, f"Formula: {formula}")
        
        self.set_font('Arial', '', 9)
        result_text = f"Result: {result}"
        if unit:
            result_text += f" {unit}"
        self.cell(0, 5, result_text, 0, 1)
        
        if details:
            self.set_font('Arial', 'I', 8)
            for key, value in details.items():
                self.cell(0, 4, f"  {key}: {value}", 0, 1)
        
        self.ln(2)
    
    def add_iteration_data(self, iteration_num, data):
        """Add iteration data"""
        self.set_font('Arial', 'B', 10)
        self.cell(0, 7, f"Iteration {iteration_num}:", 0, 1)
        
        self.set_font('Arial', '', 8)
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.cell(0, 4, f"  {key}: {value}", 0, 1)
            elif isinstance(value, dict):
                self.cell(0, 4, f"  {key}:", 0, 1)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.cell(10, 4, "", 0, 0)
                        self.cell(0, 4, f"    {subkey}: {subvalue}", 0, 1)
        self.ln(3)
    
    def add_fp_details(self, fp_details):
        """Add Fp calculation details"""
        self.add_subtitle("Piping Geometry Factor (Fp) Calculation Details")
        
        if 'note' in fp_details and fp_details['note']:
            self.add_paragraph(f"Note: {fp_details['note']}")
        
        self.set_font('Arial', 'B', 9)
        self.cell(0, 6, "Input Parameters:", 0, 1)
        self.set_font('Arial', '', 8)
        self.cell(0, 4, f"  Valve diameter: {fp_details.get('valve_d_inch', 'N/A')} inch", 0, 1)
        self.cell(0, 4, f"  Inlet pipe diameter: {fp_details.get('pipe_d_in_inch', 'N/A')} inch", 0, 1)
        self.cell(0, 4, f"  Outlet pipe diameter: {fp_details.get('pipe_d_out_inch', 'N/A')} inch", 0, 1)
        self.cell(0, 4, f"  Cv at operating point: {fp_details.get('cv_op', 'N/A'):.1f}", 0, 1)
        
        self.ln(2)
        self.set_font('Arial', 'B', 9)
        self.cell(0, 6, "Intermediate Calculations:", 0, 1)
        self.set_font('Arial', '', 8)
        self.cell(0, 4, f"  d_ratio_in (valve/inlet): {fp_details.get('d_ratio_in', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  d_ratio_out (valve/outlet): {fp_details.get('d_ratio_out', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  K1 (inlet reducer coefficient): {fp_details.get('K1', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  K2 (outlet reducer coefficient): {fp_details.get('K2', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  KB1 (inlet Bernoulli coefficient): {fp_details.get('KB1', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  KB2 (outlet Bernoulli coefficient): {fp_details.get('KB2', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  ΣK = K1 + K2 + KB1 - KB2: {fp_details.get('sumK', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  N2 constant: {fp_details.get('N2', 'N/A')}", 0, 1)
        
        self.ln(2)
        self.set_font('Arial', 'B', 9)
        self.cell(0, 6, "Final Calculation:", 0, 1)
        self.set_font('Arial', '', 8)
        if 'formula' in fp_details:
            self.cell(0, 4, f"  Formula: {fp_details['formula']}", 0, 1)
        self.cell(0, 4, f"  Term = 1 + (ΣK/N2) * (Cv/d²)²: {fp_details.get('term', 'N/A'):.4f}", 0, 1)
        self.cell(0, 4, f"  Fp = 1 / √(Term): {fp_details.get('Fp', 'N/A'):.4f}", 0, 1)
        
        self.ln(5)

def generate_detailed_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info,
                                plot_bytes=None, flow_dp_plot_bytes=None, calculation_records=None):
    """
    Generate detailed PDF report with calculation records
    """
    try:
        pdf = EnhancedPDFReport()
        pdf.add_page()
        
        # Cover section
        pdf.add_section_title("VALVE SIZING REPORT WITH CALCULATION DETAILS")
        pdf.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        pdf.add_paragraph(f"Valve: {get_valve_display_name(valve)}")
        pdf.ln(10)
        
        # Valve specifications
        pdf.add_subtitle("Valve Specifications")
        specs_data = {
            "Size": {"value": valve.size, "unit": "inch"},
            "Type": {"value": "Globe" if valve.valve_type == 3 else "Axial"},
            "Rating Class": {"value": valve.rating_class},
            "Max Cv": {"value": f"{valve.get_cv_at_opening(100):.1f}"},
            "Max Kv": {"value": f"{cv_to_kv(valve.get_cv_at_opening(100)):.1f}"},
            "Fl at 100%": {"value": f"{valve.get_fl_at_opening(100):.3f}"},
            "Xt at 100%": {"value": f"{valve.get_xt_at_opening(100):.3f}"},
            "Fd": {"value": f"{valve.fd:.2f}"},
            "Diameter": {"value": f"{valve.diameter:.2f}", "unit": "inch"}
        }
        pdf.add_key_value_table(specs_data)
        
        # Results summary
        pdf.add_subtitle("Sizing Results Summary")
        
        results_headers = ['Scenario', 'Req Cv/Kv', 'Opening%', 'Actual Cv/Kv', 'Margin%', 'Status']
        results_data = []
        
        for i, scenario in enumerate(scenarios):
            actual_cv = valve.get_cv_at_opening(op_points[i])
            actual_kv = cv_to_kv(actual_cv)
            req_kv = cv_to_kv(req_cvs[i])
            margin = (actual_cv / req_cvs[i] - 1) * 100 if req_cvs[i] > 0 else 0
            
            status = "OK"
            if "Severe" in cavitation_info[i] or "Choked" in cavitation_info[i]:
                status = "CRITICAL"
            elif "Insufficient" in warnings[i]:
                status = "UNDERSIZED"
            elif "High opening" in warnings[i]:
                status = "HIGH OPEN"
            elif "Low opening" in warnings[i]:
                status = "LOW OPEN"
            elif "High velocity" in warnings[i]:
                status = "HIGH VEL"
            
            results_data.append([
                scenario["name"],
                f"Cv: {req_cvs[i]:.1f}\nKv: {req_kv:.1f}",
                f"{op_points[i]:.1f}%",
                f"Cv: {actual_cv:.1f}\nKv: {actual_kv:.1f}",
                f"{margin:.1f}%",
                status
            ])
        
        # Create table
        col_widths = [30, 30, 20, 30, 20, 30]
        pdf.set_font('Arial', 'B', 9)
        for i, header in enumerate(results_headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        pdf.set_font('Arial', '', 8)
        for row in results_data:
            for i, item in enumerate(row):
                pdf.cell(col_widths[i], 6, item, 1, 0, 'C')
            pdf.ln()
        pdf.ln(5)
        
        # Detailed calculation records for each scenario
        for i, scenario in enumerate(scenarios):
            pdf.add_page()
            pdf.add_subtitle(f"Detailed Calculation Record: {scenario['name']}")
            
            # Scenario parameters
            pdf.add_subtitle("Scenario Parameters")
            scenario_data = {
                "Fluid Type": {"value": scenario['fluid_type']},
                "Flow Rate": {"value": f"{scenario['flow_display']:.2f}", "unit": scenario['flow_unit']},
                "Inlet Pressure": {"value": f"{scenario['p1_display']:.2f}", "unit": scenario['p1_unit']},
                "Outlet Pressure": {"value": f"{scenario['p2_display']:.2f}", "unit": scenario['p2_unit']},
                "Temperature": {"value": f"{scenario['temp_display']:.1f}", "unit": scenario['temp_unit']},
            }
            pdf.add_key_value_table(scenario_data)
            
            # Add Fp calculation details if available
            if calculation_records and i < len(calculation_records) and calculation_records[i]:
                record = calculation_records[i]
                summary = record.get_summary()
                
                # Fp Calculation Details
                if 'fp_details' in summary and summary['fp_details']:
                    pdf.add_fp_details(summary['fp_details'])
                
                # Input Parameters
                pdf.add_subtitle("Input Parameters")
                pdf.add_key_value_table(summary.get('input_parameters', {}))
                
                # Formulas Used
                if summary.get('formulas'):
                    pdf.add_subtitle("Formulas Used")
                    for formula in summary['formulas']:
                        pdf.set_font('Arial', 'B', 9)
                        pdf.cell(0, 6, formula['name'], 0, 1)
                        
                        pdf.set_font('Courier', '', 8)
                        pdf.multi_cell(0, 4, f"Formula: {formula['formula']}")
                        
                        pdf.set_font('Arial', '', 8)
                        pdf.multi_cell(0, 4, f"Explanation: {formula['explanation']}")
                        
                        if formula.get('variables'):
                            pdf.set_font('Arial', 'I', 8)
                            vars_text = "Variables: " + ", ".join([f"{k}: {v}" for k, v in formula['variables'].items()])
                            pdf.multi_cell(0, 4, vars_text)
                        
                        pdf.ln(2)
                
                # Calculation Steps
                if summary.get('steps'):
                    pdf.add_subtitle("Calculation Steps")
                    for step in summary['steps']:
                        details = step.get('details', {})
                        pdf.add_calculation_step(
                            step['step'],
                            step['description'],
                            step['formula'],
                            step['result'],
                            step['unit'],
                            details
                        )
                
                # Iterations
                if summary.get('iterations'):
                    pdf.add_subtitle("Iteration History")
                    for iteration in summary['iterations']:
                        pdf.add_iteration_data(iteration['iteration'], iteration['data'])
                
                # Intermediate Values
                if summary.get('intermediate_values'):
                    pdf.add_subtitle("Intermediate Values")
                    pdf.add_key_value_table(summary['intermediate_values'])
                
                # Assumptions
                if summary.get('assumptions'):
                    pdf.add_subtitle("Assumptions")
                    for assumption in summary['assumptions']:
                        pdf.set_font('Arial', '', 9)
                        pdf.cell(0, 5, f"• {assumption['assumption']}", 0, 1)
                        pdf.set_font('Arial', 'I', 8)
                        pdf.multi_cell(0, 4, f"  Reason: {assumption['reason']}")
                        pdf.ln(1)
                
                # Warnings
                if summary.get('warnings'):
                    pdf.add_subtitle("Warnings and Notes")
                    for warning in summary['warnings']:
                        severity = warning['severity'].upper()
                        pdf.set_font('Arial', 'B' if severity == 'ERROR' else '', 9)
                        pdf.cell(0, 5, f"[{severity}] {warning['warning']}", 0, 1)
                        pdf.ln(1)
                
                # Final Results
                if summary.get('results'):
                    pdf.add_subtitle("Final Results")
                    pdf.add_key_value_table(summary['results'])
        
        # Add Cv curve if available
        if plot_bytes:
            pdf.add_page()
            pdf.add_subtitle("Valve Cv Characteristic")
            # Note: add_image_safe method needs to be implemented or we need to handle image differently
            # For now, we'll skip the image or add a placeholder
            
        # Add flow vs pressure drop if available
        if flow_dp_plot_bytes and scenarios:
            pdf.add_page()
            pdf.add_subtitle("Flow vs Pressure Drop")
            # Note: Same image handling issue
        
        # Final notes
        pdf.add_page()
        pdf.add_subtitle("Calculation Notes")
        pdf.add_paragraph("This report was generated by VASTAS Valve Sizing Software.")
        pdf.add_paragraph("Calculations based on ISA-75.01.01 / IEC 60534-2-1 standards.")
        pdf.add_paragraph("Pipe sizing corrections applied using separate inlet/outlet diameters.")
        pdf.add_paragraph("All values are for engineering reference only.")
        pdf.add_paragraph("Calculation records show step-by-step process with actual values used.")
        
        pdf_bytes_io = BytesIO()
        pdf.output(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        return pdf_bytes_io
        
    except Exception as e:
        # Fallback to simple PDF
        return generate_simple_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info,
                                         plot_bytes, flow_dp_plot_bytes, calculation_records)

def generate_simple_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info,
                              plot_bytes=None, flow_dp_plot_bytes=None, calculation_records=None):
    """Simple fallback PDF generation"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Valve Sizing Results', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Valve: {get_valve_display_name(valve)}', 0, 1)
        pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
        pdf.ln(10)
        
        for i, scenario in enumerate(scenarios):
            if i < len(op_points) and i < len(req_cvs):
                actual_cv = valve.get_cv_at_opening(op_points[i])
                actual_kv = cv_to_kv(actual_cv)
                req_kv = cv_to_kv(req_cvs[i])
                margin = (actual_cv / req_cvs[i] - 1) * 100 if req_cvs[i] > 0 else 0
                pdf.cell(0, 6, f"{scenario['name']}: ReqCv={req_cvs[i]:.1f}(Kv={req_kv:.1f}), Open={op_points[i]:.1f}%, Margin={margin:.1f}%", 0, 1)
        
        pdf_bytes_io = BytesIO()
        pdf.output(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        return pdf_bytes_io
    except:
        # Ultimate fallback
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Valve Sizing Report', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, 'Calculation completed successfully.', 0, 1)
        
        pdf_bytes_io = BytesIO()
        pdf.output(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        return pdf_bytes_io

def generate_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info,
                       plot_bytes=None, flow_dp_plot_bytes=None, logo_bytes=None, 
                       logo_type=None, client_info=None, project_notes=None,
                       calculation_records=None):
    """
    Main PDF generation function - uses detailed version
    """
    return generate_detailed_pdf_report(
        scenarios, valve, op_points, req_cvs, warnings, cavitation_info,
        plot_bytes, flow_dp_plot_bytes, calculation_records
    )

# ========================
# SIMULATION RESULTS
# ========================
def get_simulation_image(valve_name):
    simulation_images = {
        "2.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/2e33.png",
        "4.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/4e33.png",
        "8.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/8e33.png",
        "8.0\" E43": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/8e43.png",
        "12.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/12e33.png",
        "16.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/16e33.png",
        "20.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/20e33.png",
        "30.0\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/30e33.png",
    }
    return simulation_images.get(valve_name, "https://via.placeholder.com/1200x900.png?text=Simulation+Not+Available")

# ========================
# FLOW RATE VS PRESSURE DROP GRAPH
# ========================
def generate_flow_vs_dp_graph(scenario, valve, op_point, details, req_cv):
    # Get actual Cv at operating point
    actual_cv = valve.get_cv_at_opening(op_point)
    valve_cv_effective = actual_cv * details.get('fp', 1.0)
    
    # Determine max pressure drop
    if scenario['fluid_type'] == "liquid":
        max_dp = details.get('dp_max', scenario['p1'] - scenario['p2'])
    elif scenario['fluid_type'] in ["gas", "steam"]:
        x_crit = details.get('x_crit', 0)
        if x_crit <= 0:
            k = scenario.get('k', 1.4)
            xt = details.get('xt_at_op', 0.5)
            fk = k / 1.4
            x_crit = fk * xt
        max_dp = x_crit * scenario['p1']
    else:
        max_dp = scenario['p1'] - scenario['p2']
    
    # Create pressure drop range (from 1/10 max to max)
    min_dp = max(0.1, max_dp / 10)  # Ensure min_dp is at least 0.1 bar
    dp_range = np.linspace(min_dp, max_dp, 200)
    flow_rates = []
    
    # Calculate flow rates for each dp
    for dp in dp_range:
        if scenario["fluid_type"] == "liquid":
            if dp <= details.get('dp_max', dp):
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * math.sqrt(dp / scenario['sg'])
            else:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * details.get('fl_at_op', 0.9) * math.sqrt(
                    (scenario['p1'] - details.get('ff', 0.96) * scenario.get('pv', 0)) / scenario['sg'])
            flow_rates.append(flow)
            
        elif scenario["fluid_type"] == "gas":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N7"]["m³/h, bar, K (standard)"] * scenario['p1'] * Y * math.sqrt(
                x / (scenario['sg'] * (scenario['temp'] + C_TO_K) * scenario['z']))
            flow_rates.append(flow)
            
        elif scenario["fluid_type"] == "steam":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N6"]["kg/h, bar, kg/m³"] * Y * math.sqrt(
                x * scenario['p1'] * scenario['rho'])
            flow_rates.append(flow)
        else:
            flow_rates.append(0)
    
    # Current operating point
    current_dp = scenario['p1'] - scenario['p2']
    current_flow = scenario['flow']
    
    # Create smooth curve using polynomial interpolation
    if len(dp_range) > 3 and len(flow_rates) > 3:
        z = np.polyfit(dp_range, flow_rates, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min_dp, max_dp, 300)
        y_smooth = p(x_smooth)
    else:
        x_smooth = dp_range
        y_smooth = flow_rates
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_smooth, 
        y=y_smooth, 
        mode='lines',
        name='Flow Rate',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[current_dp], 
        y=[current_flow], 
        mode='markers',
        name='Operating Point',
        marker=dict(size=12, color='red')
    ))
    
    # Add max flow annotation
    if max_dp > 0 and flow_rates:
        max_flow = flow_rates[-1]
        fig.add_annotation(
            x=max_dp,
            y=max_flow,
            text=f'Max Flow: {max_flow:.1f}',
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-30
        )
    
    # Determine y-axis label
    if scenario["fluid_type"] == "liquid":
        y_axis_label = 'Flow Rate (m³/h)'
    elif scenario["fluid_type"] == "gas":
        y_axis_label = 'Flow Rate (std m³/h)'
    else:
        y_axis_label = 'Flow Rate (kg/h)'
    
    fig.update_layout(
        title=f'Flow Rate vs Pressure Drop - {scenario["name"]}',
        xaxis_title='Pressure Drop (bar)',
        yaxis_title=y_axis_label,
        legend_title='Legend',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    fig.update_xaxes(range=[0, max_dp * 1.1])
    if flow_rates:
        fig.update_yaxes(range=[0, max(flow_rates) * 1.1])
    
    return fig

# ========================
# MATPLOTLIB PLOT FOR PDF
# ========================
def plot_cv_curve_matplotlib(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    openings = sorted(valve.cv_table.keys())
    cv_values = np.array([valve.get_cv_at_opening(op) for op in openings])
    kv_values = np.array([cv_to_kv(cv) for cv in cv_values])
    
    # Create dense x values for smooth curve
    x_smooth = np.linspace(0, 100, 300)
    
    # Create polynomial fit
    if len(openings) > 3:
        z = np.polyfit(openings, cv_values, 3)
        p = np.poly1d(z)
        cv_smooth = p(x_smooth)
        kv_smooth = cv_to_kv(cv_smooth)
    else:
        cv_smooth = np.interp(x_smooth, openings, cv_values)
        kv_smooth = np.interp(x_smooth, openings, kv_values)
    
    plt.figure(figsize=(10, 6))
    
    # Valve Cv curve
    plt.plot(x_smooth, cv_smooth, 'b-', linewidth=2, label='Valve Cv')
    plt.plot(x_smooth, kv_smooth, 'g-', linewidth=2, label='Valve Kv', alpha=0.7)
    
    # Operating points
    for i, op in enumerate(op_points):
        actual_cv = valve.get_cv_at_opening(op)
        actual_kv = cv_to_kv(actual_cv)
        plt.plot(op, actual_cv, 'ro', markersize=8)
        plt.plot(op, actual_kv, 'go', markersize=6, alpha=0.7)
        plt.text(op + 2, actual_cv, f'S{i+1}', fontsize=10, color='red')
    
    # Required Cv lines
    for i, cv in enumerate(req_cvs):
        kv_req = cv_to_kv(cv)
        plt.axhline(y=cv, color='r', linestyle='--', linewidth=1)
        plt.axhline(y=kv_req, color='g', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(100, cv, f'Corrected S{i+1} Cv: {cv:.1f}', 
                 fontsize=9, color='red', ha='right', va='bottom')
        plt.text(100, kv_req, f'Corrected S{i+1} Kv: {kv_req:.1f}', 
                 fontsize=9, color='green', ha='right', va='top', alpha=0.7)
    
    # Theoretical Cv lines
    for i, cv in enumerate(theoretical_cvs):
        kv_theoretical = cv_to_kv(cv)
        plt.axhline(y=cv, color='r', linestyle=':', linewidth=1)
        plt.axhline(y=kv_theoretical, color='g', linestyle=':', linewidth=1, alpha=0.7)
        plt.text(100, cv, f'Theoretical S{i+1} Cv: {cv:.1f}', 
                 fontsize=9, color='red', ha='right', va='top')
        plt.text(100, kv_theoretical, f'Theoretical S{i+1} Kv: {kv_theoretical:.1f}', 
                 fontsize=9, color='green', ha='right', va='bottom', alpha=0.7)
    
    plt.title(f'{valve.size}" Valve Cv & Kv Characteristic')
    plt.xlabel('Opening Percentage (%)')
    plt.ylabel('Cv/Kv Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.xlim(0, 100)
    plt.ylim(0, max(cv_values) * 1.1)
    
    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def plot_flow_vs_dp_matplotlib(scenario, valve, op_point, details, req_cv):
    actual_cv = valve.get_cv_at_opening(op_point)
    valve_cv_effective = actual_cv * details.get('fp', 1.0)
    
    # Determine max pressure drop
    if scenario['fluid_type'] == "liquid":
        max_dp = details.get('dp_max', scenario['p1'] - scenario['p2'])
    elif scenario['fluid_type'] in ["gas", "steam"]:
        x_crit = details.get('x_crit', 0)
        if x_crit <= 0:
            k = scenario.get('k', 1.4)
            xt = details.get('xt_at_op', 0.5)
            fk = k / 1.4
            x_crit = fk * xt
        max_dp = x_crit * scenario['p1']
    else:
        max_dp = scenario['p1'] - scenario['p2']
    
    # Create pressure drop range
    min_dp = max(0.1, max_dp / 10)
    dp_range = np.linspace(min_dp, max_dp, 200)
    flow_rates = []
    
    # Calculate flow rates
    for dp in dp_range:
        if scenario['fluid_type'] == "liquid":
            if dp <= details.get('dp_max', dp):
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * math.sqrt(dp / scenario['sg'])
            else:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * details.get('fl_at_op', 0.9) * math.sqrt(
                    (scenario['p1'] - details.get('ff', 0.96) * scenario.get('pv', 0)) / scenario['sg'])
            flow_rates.append(flow)
        elif scenario['fluid_type'] == "gas":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N7"]["m³/h, bar, K (standard)"] * scenario['p1'] * Y * math.sqrt(
                x / (scenario['sg'] * (scenario['temp'] + C_TO_K) * scenario['z']))
            flow_rates.append(flow)
        elif scenario['fluid_type'] == "steam":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N6"]["kg/h, bar, kg/m³"] * Y * math.sqrt(
                x * scenario['p1'] * scenario['rho'])
            flow_rates.append(flow)
        else:
            flow_rates.append(0)
    
    # Current operating point
    current_dp = scenario['p1'] - scenario['p2']
    current_flow = scenario['flow']
    
    # Create smooth curve
    if len(dp_range) > 3 and len(flow_rates) > 3:
        z = np.polyfit(dp_range, flow_rates, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min_dp, max_dp, 300)
        y_smooth = p(x_smooth)
    else:
        x_smooth = dp_range
        y_smooth = flow_rates
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Flow Rate')
    plt.plot(current_dp, current_flow, 'ro', markersize=8, label='Operating Point')
    
    # Mark max flow
    if flow_rates:
        plt.plot(max_dp, flow_rates[-1], 'go', markersize=8)
        plt.annotate(f'Max Flow: {flow_rates[-1]:.1f}', 
                     xy=(max_dp, flow_rates[-1]), 
                     xytext=(-50, -30), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='green'))
    
    # Determine y-axis label
    if scenario["fluid_type"] == "liquid":
        y_axis_label = 'Flow Rate (m³/h)'
    elif scenario["fluid_type"] == "gas":
        y_axis_label = 'Flow Rate (std m³/h)'
    else:
        y_axis_label = 'Flow Rate (kg/h)'
    
    plt.title(f'Flow Rate vs Pressure Drop - {scenario["name"]}')
    plt.xlabel('Pressure Drop (bar)')
    plt.ylabel(y_axis_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xlim(0, max_dp * 1.1)
    if flow_rates:
        plt.ylim(0, max(flow_rates) * 1.1)
    
    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

# ========================
# UPDATED EVALUATE_VALVE_FOR_SCENARIO WITH ENHANCED CALCULATION RECORD
# ========================
def evaluate_valve_for_scenario_with_record(valve, scenario):
    """
    Evaluate valve for scenario with enhanced calculation record
    """
    # Initialize calculation record
    record = CalculationRecord(scenario['name'], scenario['fluid_type'])
    
    # Store all input parameters with original units
    for key, value in scenario.items():
        if isinstance(value, (int, float, str)):
            original_value = scenario.get(f'{key}_display', value)
            original_unit = scenario.get(f'{key}_unit', '')
            record.set_input_parameter(key.replace('_', ' ').title(), value, "", original_value, original_unit)
    
    # Get pipe diameters
    use_valve_size = scenario.get("use_valve_size", True)
    if use_valve_size:
        pipe_d_in = valve.diameter
        pipe_d_out = valve.diameter
    else:
        pipe_d_in = scenario["pipe_d_in"]
        pipe_d_out = scenario["pipe_d_out"]
    
    valve_d = valve.diameter
    fluid_type = scenario["fluid_type"]
    flow_std = scenario["flow"]
    
    if record:
        record.add_step(1, "Get pipe dimensions", 
                       f"Inlet: {pipe_d_in}\", Outlet: {pipe_d_out}\", Valve: {valve_d}\"", 
                       "Pipe dimensions set", "", 
                       {"Inlet pipe": f"{pipe_d_in}\"", "Outlet pipe": f"{pipe_d_out}\"", "Valve": f"{valve_d}\""})
        record.add_assumption(f"Inlet pipe diameter: {pipe_d_in}\"", 
                             "User specified or valve size",
                             {"use_valve_size": use_valve_size, "pipe_d_in": pipe_d_in})
        record.add_assumption(f"Outlet pipe diameter: {pipe_d_out}\"", 
                             "User specified or valve size",
                             {"use_valve_size": use_valve_size, "pipe_d_out": pipe_d_out})
    
    # Initialize iteration variables
    max_iterations = 20
    tolerance = 0.1
    prev_opening = 0
    opening = 50
    converged = False
    iteration_results = []
    
    for iteration in range(max_iterations):
        # Record iteration start
        if record:
            record.add_step(2 + iteration, f"Iteration {iteration + 1} - Start", 
                           f"Current opening: {opening}%", 
                           "Beginning iteration", "",
                           {"Iteration": iteration + 1, "Opening": opening})
        
        # Get ALL valve characteristics at CURRENT opening
        cv_at_op = valve.get_cv_at_opening(opening)
        fl_at_op = valve.get_fl_at_opening(opening)
        xt_at_op = valve.get_xt_at_opening(opening)
        
        if record and iteration == 0:
            record.add_step(3 + iteration, "Get valve characteristics at initial opening", 
                           f"Opening: {opening}%, Cv: {cv_at_op:.1f}, Fl: {fl_at_op:.3f}, xT: {xt_at_op:.4f}", 
                           "Valve characteristics retrieved", "",
                           {"Opening": opening, "Cv": cv_at_op, "Fl": fl_at_op, "xT": xt_at_op})
        
        # Calculate Fp using separate inlet/outlet pipe diameters with details
        fp, fp_details = calculate_piping_factor_fp_separate(valve_d, pipe_d_in, pipe_d_out, cv_at_op, record)
        
        if record and iteration == 0:
            record.set_fp_details(fp_details)
            record.add_step(4 + iteration, "Calculate piping geometry factor Fp", 
                           "Fp = 1 / √(1 + (ΣK / N2) * (Cv/d²)²)", 
                           f"{fp:.4f}", "",
                           {"Fp": fp, "Cv": cv_at_op, "Valve_d": valve_d, "Pipe_d_in": pipe_d_in, "Pipe_d_out": pipe_d_out})
        
        # Apply piping corrections if pipe diameter differs
        fl_effective = fl_at_op
        xt_effective = xt_at_op
        
        if not use_valve_size:
            if fluid_type == "liquid":
                fl_effective, flp_details = calculate_flp_separate(valve, valve_d, pipe_d_in, pipe_d_out, cv_at_op, fl_at_op, record)
                if record and iteration == 0:
                    record.add_step(5 + iteration, "Calculate combined liquid pressure recovery factor FLP", 
                                   "FLP = 1 / √((Ki/N2)*(Cv/d²)² + 1/Fl²)", 
                                   f"{fl_effective:.4f}", "",
                                   {"FLP": fl_effective, "Ki": "Inlet loss coefficient", "Cv": cv_at_op})
            elif fluid_type in ["gas", "steam"]:
                xt_effective, xtp_details = calculate_x_tp_separate(valve, valve_d, pipe_d_in, pipe_d_out, fp, cv_at_op, xt_at_op, record)
                if record and iteration == 0:
                    record.add_step(5 + iteration, "Calculate combined pressure drop ratio factor xTP", 
                                   "xTP = xT / Fp² * (1 / (1 + (xT * Ki / N5) * (Cv/d²)²))", 
                                   f"{xt_effective:.4f}", "",
                                   {"xTP": xt_effective, "xT": xt_at_op, "Fp": fp, "Cv": cv_at_op})
        
        # Calculate required Cv with updated factors
        if fluid_type == "liquid":
            if scenario.get('fluid_library') in FLUID_LIBRARY:
                fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
                scenario["visc"] = fluid_data["visc_func"](scenario["temp"], scenario["p1"])
                scenario["pv"] = fluid_data["pv_func"](scenario["temp"], scenario["p1"])
                if "pc_func" in fluid_data:
                    scenario["pc"] = fluid_data["pc_func"]()
            
            cv_req, details = cv_liquid_with_record(
                flow=flow_std,
                p1=scenario["p1"],
                p2=scenario["p2"],
                sg=scenario["sg"],
                fl_at_op=fl_effective,
                pv=scenario["pv"],
                pc=scenario["pc"],
                visc_cst=scenario["visc"],
                d_m=valve.diameter * 0.0254,
                valve=valve,
                fp=fp,
                cv_op=cv_at_op,
                record=record if iteration == 0 else None
            )
            
        elif fluid_type == "gas":
            if scenario.get('fluid_library') in FLUID_LIBRARY:
                fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
                scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
                if "z_func" in fluid_data:
                    scenario["z"] = fluid_data["z_func"](scenario["temp"], scenario["p1"])
            
            cv_req, details = cv_gas_with_record(
                flow=flow_std,
                p1=scenario["p1"],
                p2=scenario["p2"],
                sg=scenario["sg"],
                t=scenario["temp"],
                k=scenario["k"],
                xt_at_op=xt_effective,
                z=scenario["z"],
                fp=fp,
                op_point=opening,
                cv_op=cv_at_op,
                record=record if iteration == 0 else None
            )
            
        else:  # steam
            if scenario.get('fluid_library') in FLUID_LIBRARY:
                fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
                scenario["rho"] = fluid_data["rho_func"](scenario["temp"], scenario["p1"])
                scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
            
            cv_req, details = cv_steam_with_record(
                flow=flow_std,
                p1=scenario["p1"],
                p2=scenario["p2"],
                rho=scenario["rho"],
                k=scenario["k"],
                xt_at_op=xt_effective,
                fp=fp,
                op_point=opening,
                cv_op=cv_at_op,
                record=record if iteration == 0 else None
            )
        
        # Find new opening that provides at least the required Cv
        new_opening = 0
        cv_valve = 0
        
        # Binary search for optimal opening
        low, high = 0, 100
        for search_iter in range(30):
            mid = (low + high) / 2
            cv_mid = valve.get_cv_at_opening(mid)
            if cv_mid >= cv_req:
                high = mid
                cv_valve = cv_mid
                new_opening = mid
            else:
                low = mid
        
        # If binary search didn't find exact, do linear search
        if new_opening == 0 or cv_valve < cv_req:
            new_opening = 0
            while new_opening <= 100:
                cv_valve = valve.get_cv_at_opening(new_opening)
                if cv_valve >= cv_req:
                    break
                new_opening += 1
        
        # Store iteration results for display
        iteration_data = {
            'iteration': iteration + 1,
            'opening': opening,
            'cv_at_op': cv_at_op,
            'kv_at_op': cv_to_kv(cv_at_op),
            'fl_at_op': fl_at_op,
            'xt_at_op': xt_at_op,
            'fl_effective': fl_effective if fluid_type == "liquid" else None,
            'xt_effective': xt_effective if fluid_type in ["gas", "steam"] else None,
            'fp': fp,
            'fp_details': fp_details,  # Include Fp details in iteration data
            'cv_req': cv_req,
            'kv_req': cv_to_kv(cv_req),
            'new_opening': new_opening,
            'convergence_diff': abs(new_opening - opening)
        }
        
        iteration_results.append(iteration_data)
        
        # Add iteration to record
        if record:
            record.add_iteration(iteration + 1, iteration_data)
            record.add_step(6 + iteration, f"Iteration {iteration + 1} - Results", 
                           f"Opening: {opening}% → {new_opening}%, Req Cv: {cv_req:.1f}", 
                           "Iteration completed", "",
                           {"Old opening": opening, "New opening": new_opening, "Required Cv": cv_req})
        
        # Check for convergence
        if abs(new_opening - opening) < tolerance:
            opening = new_opening
            converged = True
            if record:
                record.add_step(7 + iteration, f"Iteration {iteration + 1} - Convergence", 
                               f"|{new_opening} - {opening}| < {tolerance}", 
                               "Converged successfully", "",
                               {"Difference": abs(new_opening - opening), "Tolerance": tolerance})
            break
        
        # Update for next iteration
        opening = new_opening
        
        # Early exit if at boundary
        if opening >= 100 or opening <= 0:
            converged = True
            if record:
                record.add_step(7 + iteration, f"Iteration {iteration + 1} - Boundary reached", 
                               f"Opening at boundary: {opening}%", 
                               "Stopping iterations", "",
                               {"Opening": opening})
            break
    
    # After iteration, get FINAL values at converged opening
    final_cv = valve.get_cv_at_opening(opening)
    final_fl = valve.get_fl_at_opening(opening)
    final_xt = valve.get_xt_at_opening(opening)
    
    # Final Fp calculation with converged Cv
    final_fp, final_fp_details = calculate_piping_factor_fp_separate(valve_d, pipe_d_in, pipe_d_out, final_cv, record)
    
    # Apply final piping corrections if needed
    final_fl_effective = final_fl
    final_xt_effective = final_xt
    
    if not use_valve_size:
        if fluid_type == "liquid":
            final_fl_effective, _ = calculate_flp_separate(valve, valve_d, pipe_d_in, pipe_d_out, final_cv, final_fl, record)
        elif fluid_type in ["gas", "steam"]:
            final_xt_effective, _ = calculate_x_tp_separate(valve, valve_d, pipe_d_in, pipe_d_out, final_fp, final_cv, final_xt, record)
    
    # FINAL Cv calculation with all converged values
    if fluid_type == "liquid":
        cv_req_final, details_final = cv_liquid_with_record(
            flow=flow_std,
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            fl_at_op=final_fl_effective,
            pv=scenario["pv"],
            pc=scenario["pc"],
            visc_cst=scenario["visc"],
            d_m=valve.diameter * 0.0254,
            valve=valve,
            fp=final_fp,
            cv_op=final_cv,
            record=None  # Don't record final iteration details
        )
        
        if scenario["pc"] > 0:
            choked, sigma, km, cav_msg = check_cavitation_with_record(
                scenario["p1"], scenario["p2"], scenario["pv"], final_fl_effective, scenario["pc"], record
            )
            details_final['is_choked'] = choked
            details_final['cavitation_severity'] = cav_msg
            details_final['sigma'] = sigma
            details_final['km'] = km
        else:
            details_final['cavitation_severity'] = "Critical pressure not available"
            
    elif fluid_type == "gas":
        cv_req_final, details_final = cv_gas_with_record(
            flow=flow_std,
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            t=scenario["temp"],
            k=scenario["k"],
            xt_at_op=final_xt_effective,
            z=scenario["z"],
            fp=final_fp,
            op_point=opening,
            cv_op=final_cv,
            record=None
        )
        
        if details_final.get('is_choked', False):
            details_final['cavitation_severity'] = "Choked flow detected"
        else:
            details_final['cavitation_severity'] = "No choked flow"
            
    else:  # steam
        cv_req_final, details_final = cv_steam_with_record(
            flow=flow_std,
            p1=scenario["p1"],
            p2=scenario["p2"],
            rho=scenario["rho"],
            k=scenario["k"],
            xt_at_op=final_xt_effective,
            fp=final_fp,
            op_point=opening,
            cv_op=final_cv,
            record=None
        )
        
        if details_final.get('is_choked', False):
            details_final['cavitation_severity'] = "Choked flow detected"
        else:
            details_final['cavitation_severity'] = "No choked flow"
    
    # ADD THE FINAL ITERATION TO THE RESULTS
    final_iteration_data = {
        'iteration': len(iteration_results) + 1,
        'opening': opening,
        'cv_at_op': final_cv,
        'kv_at_op': cv_to_kv(final_cv),
        'fl_at_op': final_fl,
        'xt_at_op': final_xt,
        'fl_effective': final_fl_effective if fluid_type == "liquid" else None,
        'xt_effective': final_xt_effective if fluid_type in ["gas", "steam"] else None,
        'fp': final_fp,
        'fp_details': final_fp_details,  # Include final Fp details
        'cv_req': cv_req_final,
        'kv_req': cv_to_kv(cv_req_final),
        'new_opening': opening,
        'convergence_diff': 0,
        'note': 'FINAL ITERATION - ALL CORRECTIONS APPLIED'
    }
    
    iteration_results.append(final_iteration_data)
    
    if record:
        record.add_iteration(len(iteration_results), final_iteration_data)
        record.add_step(100, "Final iteration completed", 
                       f"Final opening: {opening}%, Final Cv: {final_cv:.1f}, Required Cv: {cv_req_final:.1f}", 
                       "All calculations completed", "",
                       {"Final opening": opening, "Final Cv": final_cv, "Required Cv": cv_req_final})
        record.set_fp_details(final_fp_details)  # Store final Fp details
    
    # Calculate velocities with corrected pressure-dependent volumetric flow
    velocity_results = calculate_valve_velocity_separate(scenario, valve, opening)
    
    # Record velocity calculations
    if record:
        record.add_step(101, "Calculate flow velocities", 
                       f"Inlet flow: {velocity_results['inlet_flow_m3h']:.2f} m³/h at {scenario['p1']} bar, Outlet flow: {velocity_results['outlet_flow_m3h']:.2f} m³/h at {scenario['p2']} bar", 
                       "Velocity calculations with pressure-dependent volumetric flow", "",
                       {"Inlet velocity": f"{velocity_results['inlet_velocity']:.2f} m/s",
                        "Outlet velocity": f"{velocity_results['outlet_velocity']:.2f} m/s",
                        "Orifice velocity": f"{velocity_results['orifice_velocity']:.2f} m/s",
                        "Inlet density": f"{velocity_results['inlet_density']:.2f} kg/m³",
                        "Outlet density": f"{velocity_results['outlet_density']:.2f} kg/m³",
                        "Flow ratio (out/in)": f"{velocity_results['flow_ratio']:.3f}"})
    
    # Get final valve Cv at opening
    cv_valve_final = valve.get_cv_at_opening(opening)
    kv_valve_final = cv_to_kv(cv_valve_final)
    
    warn = ""
    # Check for insufficient capacity
    if opening >= 100 and cv_valve_final < cv_req_final:
        warn = "Insufficient Capacity – Valve is undersized"
        status = "red"
    elif opening < 20:
        warn = "Low opening (<20%)"
        status = "yellow"
    elif opening > 80:
        warn = "High opening (>80%)"
        status = "yellow"
    else:
        status = "green"
    
    # Add velocity warning
    if velocity_results['warning']:
        if warn:
            warn += "; " + velocity_results['warning']
        else:
            warn = velocity_results['warning']
            if status == "green":
                status = "yellow"
    
    # Override status based on flow conditions
    if details_final.get('is_choked', False):
        status = "red"
        if "Insufficient" not in warn and "High velocity" not in warn:
            if warn:
                warn = "Choked flow - " + warn
            else:
                warn = "Choked flow"
    elif "Severe" in details_final.get('cavitation_severity', ""):
        status = "orange"
    elif "Moderate" in details_final.get('cavitation_severity', ""):
        status = "yellow"
    
    if record:
        record.set_result("Final Opening", opening, "%", "Valve opening percentage")
        record.set_result("Final Cv", cv_valve_final, "", "Valve Cv at operating point")
        record.set_result("Final Kv", kv_valve_final, "", "Valve Kv at operating point")
        record.set_result("Required Cv", cv_req_final, "", "Required Cv after all corrections")
        record.set_result("Required Kv", cv_to_kv(cv_req_final), "", "Required Kv after all corrections")
        record.set_result("Margin", (cv_valve_final / cv_req_final - 1) * 100 if cv_req_final > 0 else 0, "%", 
                         "Capacity margin ((Actual/Req - 1)*100)")
        record.set_result("Status", status, "", "Overall valve status")
        record.set_result("Warning", warn, "", "Warnings and issues")
        record.set_result("Cavitation Info", details_final.get('cavitation_severity', "N/A"), "", "Cavitation assessment")
        
        record.add_warning(f"Status: {status}", "info" if status == "green" else "warning",
                         {"Status": status, "Opening": opening, "Margin": (cv_valve_final / cv_req_final - 1) * 100})
        
        # Store additional results
        record.set_result("Orifice Velocity", velocity_results['orifice_velocity'], "m/s", "Velocity through valve orifice")
        record.set_result("Inlet Velocity", velocity_results['inlet_velocity'], "m/s", "Velocity in inlet pipe")
        record.set_result("Outlet Velocity", velocity_results['outlet_velocity'], "m/s", "Velocity in outlet pipe")
        record.set_result("Inlet ΔP (velocity)", velocity_results['inlet_pressure_drop'], "bar", "Velocity pressure drop in inlet")
        record.set_result("Outlet ΔP (velocity)", velocity_results['outlet_pressure_drop'], "bar", "Velocity pressure drop in outlet")
        record.set_result("Orifice ΔP (velocity)", velocity_results.get('orifice_pressure_drop', 0), "bar", "Velocity pressure drop at orifice")
        record.set_result("Fp (final)", final_fp, "", "Final piping geometry factor")
        record.set_result("Fl effective", final_fl_effective if fluid_type == "liquid" else "N/A", "", "Effective liquid recovery factor")
        record.set_result("xT effective", final_xt_effective if fluid_type in ["gas", "steam"] else "N/A", "", "Effective pressure drop ratio factor")
        record.set_result("Converged", converged, "", "Iteration convergence status")
        record.set_result("Iterations used", len(iteration_results), "", "Number of iterations performed")
        record.set_result("Inlet Density", velocity_results.get('inlet_density', 0), "kg/m³", "Density at inlet conditions")
        record.set_result("Outlet Density", velocity_results.get('outlet_density', 0), "kg/m³", "Density at outlet conditions")
        record.set_result("Flow Ratio (out/in)", velocity_results.get('flow_ratio', 0), "", "Ratio of outlet to inlet volumetric flow")
    
    return {
        "op_point": opening,
        "req_cv": cv_req_final,
        "req_kv": cv_to_kv(cv_req_final),
        "theoretical_cv": details_final.get('theoretical_cv', 0),
        "theoretical_kv": cv_to_kv(details_final.get('theoretical_cv', 0)),
        "warning": warn,
        "cavitation_info": details_final.get('cavitation_severity', "N/A"),
        "status": status,
        "margin": (cv_valve_final / cv_req_final - 1) * 100 if cv_req_final > 0 else 0,
        "details": details_final,
        "orifice_velocity": velocity_results['orifice_velocity'],
        "inlet_velocity": velocity_results['inlet_velocity'],
        "outlet_velocity": velocity_results['outlet_velocity'],
        "inlet_pressure_drop": velocity_results['inlet_pressure_drop'],
        "outlet_pressure_drop": velocity_results['outlet_pressure_drop'],
        "orifice_pressure_drop": velocity_results.get('orifice_pressure_drop', 0),
        "fp": final_fp,
        "fp_details": final_fp_details,  # Include Fp details in results
        "fl_effective": final_fl_effective,
        "xt_effective": final_xt_effective,
        "iterations": iteration_results,
        "converged": converged,
        "calculation_record": record,
        "velocity_details": velocity_results  # Include all velocity details
    }

def find_recommended_valve(scenarios):
    best_valve = None
    best_score = float('-inf')
    all_valve_results = []
    
    for valve in VALVE_DATABASE:
        valve_results = []
        valve_score = 0
        is_suitable = True
        
        for scenario in scenarios:
            result = evaluate_valve_for_scenario_with_record(valve, scenario)
            valve_results.append(result)
            
            if result["status"] == "red":
                valve_score -= 100
                is_suitable = False
            elif result["status"] == "orange":
                valve_score -= 50
            elif result["status"] == "yellow":
                valve_score -= 10
            else:
                valve_score += 20
                
            valve_score += max(0, 10 - abs(result["op_point"] - 50)/5)
        
        valve_display_name = get_valve_display_name(valve)
        all_valve_results.append({
            "valve": valve,
            "results": valve_results,
            "score": valve_score,
            "display_name": valve_display_name
        })
        
        if is_suitable and valve_score > best_score:
            best_valve = {
                "valve": valve,
                "results": valve_results,
                "score": valve_score,
                "display_name": valve_display_name
            }
            best_score = valve_score
    
    if best_valve is None and all_valve_results:
        all_valve_results.sort(key=lambda x: x["score"], reverse=True)
        best_valve = all_valve_results[0]
    
    return best_valve, all_valve_results

# ========================
# FLUID PROPERTY UPDATE FUNCTIONS
# ========================
def update_fluid_properties(scenario_num, fluid_library, temp_c, p1_bar):
    """Update fluid properties in session state when fluid library, temperature or pressure changes"""
    if fluid_library != "Select Fluid Library...":
        if fluid_library in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[fluid_library]
            
            # Update session state with recalculated properties
            if fluid_data["type"] == "liquid":
                if fluid_data.get("visc_func"):
                    st.session_state[f"visc_{scenario_num}"] = fluid_data["visc_func"](temp_c, p1_bar)
                if fluid_data.get("pv_func"):
                    st.session_state[f"pv_{scenario_num}"] = fluid_data["pv_func"](temp_c, p1_bar)
                if fluid_data.get("pc_func"):
                    st.session_state[f"pc_{scenario_num}"] = fluid_data["pc_func"]()
                if fluid_data.get("sg") is not None:
                    st.session_state[f"sg_{scenario_num}"] = fluid_data["sg"]
            
            elif fluid_data["type"] == "gas":
                if fluid_data.get("k_func"):
                    st.session_state[f"k_{scenario_num}"] = fluid_data["k_func"](temp_c, p1_bar)
                if fluid_data.get("z_func"):
                    st.session_state[f"z_{scenario_num}"] = fluid_data["z_func"](temp_c, p1_bar)
                if fluid_data.get("sg") is not None:
                    st.session_state[f"sg_{scenario_num}"] = fluid_data["sg"]
            
            elif fluid_data["type"] == "steam":
                if fluid_data.get("rho_func"):
                    st.session_state[f"rho_{scenario_num}"] = fluid_data["rho_func"](temp_c, p1_bar)
                if fluid_data.get("k_func"):
                    st.session_state[f"k_{scenario_num}"] = fluid_data["k_func"](temp_c, p1_bar)

# ========================
# STREAMLIT APPLICATION WITH UNIT CONVERSION
# ========================
def get_valve_display_name(valve):
    rating_code_map = {
        150: 1,
        300: 2,
        600: 3,
        900: 4,
        1500: 5,
        2500: 6
    }
    rating_code = rating_code_map.get(valve.rating_class, valve.rating_class)
    
    # Base name
    base_name = f'{valve.size}" E{valve.valve_type}{rating_code}'
    
    # Note kontrolü: NaN, boş string veya geçersiz değerleri filtrele
    if hasattr(valve, 'note') and valve.note:
        # Note'u string'e çevir
        note_str = str(valve.note).strip()
        
        # Geçersiz değerleri kontrol et
        if (note_str and 
            note_str.lower() not in ['nan', 'none', 'null', ''] and
            not note_str.isspace()):
            return f'{base_name} ({note_str})'
    
    # Note yoksa veya geçersizse sadece base name'i döndür
    return base_name

def create_valve_dropdown():
    valves = sorted(st.session_state.valve_database, key=lambda v: (v.size, v.rating_class, v.valve_type))
    valve_options = {get_valve_display_name(v): v for v in valves}
    return valve_options

def create_fluid_dropdown():
    return ["Select Fluid Library..."] + list(FLUID_LIBRARY.keys())

def get_flow_rate_units(fluid_type):
    """Get available flow rate units based on fluid type"""
    if fluid_type == "liquid":
        return ["m³/h", "L/min", "L/s", "US gpm", "Imp gpm", "bbl/h"]
    elif fluid_type == "gas":
        return ["std m³/h", "scfm", "scfh", "Nm³/h", "MMSCFD", "L/min"]
    elif fluid_type == "steam":
        return ["kg/h", "lb/h", "t/h", "kg/s"]
    return ["m³/h"]  # Default

def get_pressure_units():
    """Get available pressure units"""
    return ["bar", "bar a", "kPa", "MPa", "psi", "psia", "mmHg", "inHg", "ftH2O", "mH2O", "atm", "kg/cm²"]

def get_temperature_units():
    """Get available temperature units"""
    return ["°C", "K", "°F", "°R"]

def get_length_units():
    """Get available length/diameter units"""
    return ["inch", "mm", "cm", "m", "ft"]

def get_viscosity_units():
    """Get available viscosity units"""
    return ["cSt", "m²/s", "ft²/s", "SSF", "SSU"]

def get_density_units():
    """Get available density units"""
    return ["kg/m³", "g/cm³", "lb/ft³", "lb/gal (US)", "SG"]

def scenario_input_form(scenario_num, scenario_data=None):
    default_values = {
        "use_valve_size": True,
        "sg": 1.0,
        "visc": 1.0,
        "pv": 0.023,
        "pc": 220.55,
        "k": 1.4,
        "z": 1.0,
        "rho": 1.0,
        "fluid_type": "liquid",
        "pipe_d_in": 2.0,
        "pipe_d_out": 2.0,
        "flow_unit": "m³/h",
        "p1_unit": "bar",
        "p2_unit": "bar",
        "temp_unit": "°C",
        "pipe_unit": "inch",
        "visc_unit": "cSt",
        "density_unit": "kg/m³"
    }
    
    if scenario_data is None:
        scenario_data = {
            "name": f"Scenario {scenario_num}",
            "fluid_type": "liquid",
            "flow": 10.0 if scenario_num == 1 else 50.0,
            "p1": 10.0,
            "p2": 6.0,
            "temp": 20.0,
            "pipe_d_in": 2.0,
            "pipe_d_out": 2.0,
            "flow_unit": "m³/h",
            "p1_unit": "bar",
            "p2_unit": "bar",
            "temp_unit": "°C",
            "pipe_unit": "inch",
            "visc_unit": "cSt",
            "density_unit": "kg/m³"
        }
        scenario_data = {**default_values, **scenario_data}
    else:
        for key, default in default_values.items():
            if key not in scenario_data:
                scenario_data[key] = default
    
    st.subheader(f"Scenario {scenario_num}")
    scenario_name = st.text_input("Scenario Name", value=scenario_data["name"], key=f"name_{scenario_num}")
    
    col1, col2 = st.columns(2)
    with col1:
        fluid_library = st.selectbox(
            "Fluid Library", 
            create_fluid_dropdown(), 
            key=f"fluid_library_{scenario_num}"
        )
    
    with col2:
        if fluid_library != "Select Fluid Library...":
            fluid_data = FLUID_LIBRARY[fluid_library]
            fluid_type = fluid_data["type"]
            st.text_input("Fluid Type", value=fluid_type.capitalize(), disabled=True, key=f"fluid_type_text_{scenario_num}")
        else:
            try:
                index_val = ["Liquid", "Gas", "Steam"].index(scenario_data["fluid_type"].capitalize())
            except (ValueError, AttributeError):
                index_val = 0
            fluid_type = st.selectbox(
                "Fluid Type", 
                ["Liquid", "Gas", "Steam"], 
                index=index_val,
                key=f"fluid_type_{scenario_num}"
            ).lower()
    
    col1, col2 = st.columns(2)
    with col1:
        # Flow rate input with unit selection
        flow_units = get_flow_rate_units(fluid_type)
        default_unit_index = flow_units.index(scenario_data["flow_unit"]) if scenario_data["flow_unit"] in flow_units else 0
        
        col_flow1, col_flow2 = st.columns([3, 2])
        with col_flow1:
            flow_value = st.number_input(
                "Flow Rate", 
                min_value=0.0, 
                max_value=10000000.0, 
                value=scenario_data.get("flow_display", scenario_data["flow"]), 
                step=0.1,
                key=f"flow_{scenario_num}"
            )
        with col_flow2:
            flow_unit = st.selectbox(
                "Unit",
                flow_units,
                index=default_unit_index,
                key=f"flow_unit_{scenario_num}"
            )
        
        # Display std m³/h equivalent
        # Get current temperature and pressure for conversion
        temp_val = scenario_data.get("temp_display", scenario_data["temp"])
        p1_val = scenario_data.get("p1_display", scenario_data["p1"])
        
        # Convert to standard units for display
        flow_std = UnitConverter.convert_flow_rate(flow_value, flow_unit, 
                                                   "std m³/h" if fluid_type == "gas" else "m³/h", 
                                                   fluid_type, temp_val, p1_val)
        st.caption(f"Equivalent flow rate: {flow_std:.2f} {'std m³/h' if fluid_type == 'gas' else 'm³/h'}")
        
        # Pressure inputs with unit selection
        col_p11, col_p12 = st.columns([3, 2])
        with col_p11:
            p1_value = st.number_input(
                "Inlet Pressure", 
                min_value=0.0, 
                max_value=1000.0, 
                value=scenario_data.get("p1_display", scenario_data["p1"]), 
                step=0.1,
                key=f"p1_{scenario_num}"
            )
        with col_p12:
            p1_unit = st.selectbox(
                "P1 Unit",
                get_pressure_units(),
                index=get_pressure_units().index(scenario_data["p1_unit"]) if scenario_data["p1_unit"] in get_pressure_units() else 0,
                key=f"p1_unit_{scenario_num}"
            )
        
        col_p21, col_p22 = st.columns([3, 2])
        with col_p21:
            p2_value = st.number_input(
                "Outlet Pressure", 
                min_value=0.0, 
                max_value=1000.0, 
                value=scenario_data.get("p2_display", scenario_data["p2"]), 
                step=0.1,
                key=f"p2_{scenario_num}"
            )
        with col_p22:
            p2_unit = st.selectbox(
                "P2 Unit",
                get_pressure_units(),
                index=get_pressure_units().index(scenario_data["p2_unit"]) if scenario_data["p2_unit"] in get_pressure_units() else 0,
                key=f"p2_unit_{scenario_num}"
            )
        
        # Temperature input with unit selection
        col_temp1, col_temp2 = st.columns([3, 2])
        with col_temp1:
            temp_value = st.number_input(
                "Temperature", 
                min_value=-200.0, 
                max_value=1000.0, 
                value=scenario_data.get("temp_display", scenario_data["temp"]), 
                step=1.0,
                key=f"temp_{scenario_num}"
            )
        with col_temp2:
            temp_unit = st.selectbox(
                "Temp Unit",
                get_temperature_units(),
                index=get_temperature_units().index(scenario_data["temp_unit"]) if scenario_data["temp_unit"] in get_temperature_units() else 0,
                key=f"temp_unit_{scenario_num}"
            )
    
    with col2:
        # Check if fluid library, temperature or pressure have changed and update properties
        current_fluid_library = fluid_library
        current_temp = UnitConverter.convert_temperature(temp_value, temp_unit, "°C")
        current_p1 = UnitConverter.convert_pressure(p1_value, p1_unit, "bar")
        
        # Always recalculate fluid properties when fluid library, temperature, or pressure changes
        if current_fluid_library != "Select Fluid Library...":
            update_fluid_properties(scenario_num, current_fluid_library, current_temp, current_p1)
        
        # Use session state values if available, otherwise use scenario data
        sg_value = st.session_state.get(f"sg_{scenario_num}", scenario_data["sg"])
        visc_value = st.session_state.get(f"visc_{scenario_num}", scenario_data["visc"])
        pv_value = st.session_state.get(f"pv_{scenario_num}", scenario_data["pv"])
        pc_value = st.session_state.get(f"pc_{scenario_num}", scenario_data["pc"])
        k_value = st.session_state.get(f"k_{scenario_num}", scenario_data["k"])
        z_value = st.session_state.get(f"z_{scenario_num}", scenario_data["z"])
        rho_value = st.session_state.get(f"rho_{scenario_num}", scenario_data["rho"])
        
        if fluid_type in ["liquid", "gas"]:
            sg = st.number_input(
                "Specific Gravity (water=1)" if fluid_type == "liquid" else "Specific Gravity (air=1)",
                min_value=0.01, 
                max_value=10.0, 
                value=sg_value, 
                step=0.01,
                key=f"sg_{scenario_num}",
                disabled=(current_fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "liquid":
            # Viscosity with unit selection
            col_visc1, col_visc2 = st.columns([3, 2])
            with col_visc1:
                visc = st.number_input(
                    "Viscosity", 
                    min_value=0.01, 
                    max_value=10000.0, 
                    value=visc_value, 
                    step=0.1,
                    key=f"visc_{scenario_num}",
                    disabled=(current_fluid_library != "Select Fluid Library...")
                )
            with col_visc2:
                visc_unit = st.selectbox(
                    "Visc Unit",
                    get_viscosity_units(),
                    index=get_viscosity_units().index(scenario_data["visc_unit"]) if scenario_data["visc_unit"] in get_viscosity_units() else 0,
                    key=f"visc_unit_{scenario_num}",
                    disabled=(current_fluid_library != "Select Fluid Library...")
                )
            
            # Convert viscosity to cSt for internal calculations
            visc_cst = UnitConverter.convert_viscosity(visc, visc_unit, "cSt")
            st.caption(f"Viscosity: {visc_cst:.2f} cSt")
            
            # Vapor pressure
            pv = st.number_input(
                "Vapor Pressure (bar a)", 
                min_value=0.0, 
                max_value=100.0, 
                value=pv_value, 
                step=0.0001,
                format="%.4f",
                key=f"pv_{scenario_num}",
                disabled=(current_fluid_library != "Select Fluid Library...")
            )
            
            # Critical pressure
            pc = st.number_input(
                "Critical Pressure (bar a)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=pc_value, 
                step=0.1,
                key=f"pc_{scenario_num}",
                disabled=(current_fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type in ["gas", "steam"]:
            k = st.number_input(
                "Specific Heat Ratio (k=Cp/Cv)", 
                min_value=1.0, 
                max_value=2.0, 
                value=k_value, 
                step=0.01,
                key=f"k_{scenario_num}",
                disabled=(current_fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "gas":
            if current_fluid_library != "Select Fluid Library...":
                z = st.number_input(
                    "Compressibility Factor (Z)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=z_value, 
                    step=0.01,
                    key=f"z_{scenario_num}",
                    disabled=True
                )
            else:
                z = st.number_input(
                    "Compressibility Factor (Z)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=z_value, 
                    step=0.01,
                    key=f"z_{scenario_num}",
                    disabled=False
                )
        
        if fluid_type == "steam":
            # Density with unit selection
            col_rho1, col_rho2 = st.columns([3, 2])
            with col_rho1:
                rho = st.number_input(
                    "Density", 
                    min_value=0.01, 
                    max_value=2000.0, 
                    value=rho_value, 
                    step=0.1,
                    key=f"rho_{scenario_num}",
                    disabled=(current_fluid_library != "Select Fluid Library...")
                )
            with col_rho2:
                density_unit = st.selectbox(
                    "Density Unit",
                    get_density_units(),
                    index=get_density_units().index(scenario_data["density_unit"]) if scenario_data["density_unit"] in get_density_units() else 0,
                    key=f"density_unit_{scenario_num}",
                    disabled=(current_fluid_library != "Select Fluid Library...")
                )
            
            # Convert density to kg/m³ for internal calculations
            rho_kg_m3 = UnitConverter.convert_density(rho, density_unit, "kg/m³")
            st.caption(f"Density: {rho_kg_m3:.2f} kg/m³")
        
        use_valve_size = st.checkbox(
            "Use valve size for pipe diameters?",
            value=scenario_data.get("use_valve_size", True),
            key=f"use_valve_size_{scenario_num}"
        )
        
        if not use_valve_size:
            col_pipe1, col_pipe2 = st.columns(2)
            with col_pipe1:
                # Inlet pipe diameter with unit selection
                col_in1, col_in2 = st.columns([3, 2])
                with col_in1:
                    pipe_d_in = st.number_input(
                        "Inlet Pipe Diameter", 
                        min_value=0.1, 
                        max_value=100.0, 
                        value=scenario_data.get("pipe_d_in_display", scenario_data["pipe_d_in"]), 
                        step=0.1,
                        key=f"pipe_d_in_{scenario_num}"
                    )
                with col_in2:
                    pipe_unit = st.selectbox(
                        "Pipe Unit",
                        get_length_units(),
                        index=get_length_units().index(scenario_data["pipe_unit"]) if scenario_data["pipe_unit"] in get_length_units() else 0,
                        key=f"pipe_unit_{scenario_num}"
                    )
            
            with col_pipe2:
                # Outlet pipe diameter
                pipe_d_out = st.number_input(
                    "Outlet Pipe Diameter", 
                    min_value=0.1, 
                    max_value=100.0, 
                    value=scenario_data.get("pipe_d_out_display", scenario_data["pipe_d_out"]), 
                    step=0.1,
                    key=f"pipe_d_out_{scenario_num}"
                )
            
            # Convert pipe diameters to inches for internal calculations
            pipe_d_in_inch = UnitConverter.convert_length(pipe_d_in, pipe_unit, "inch")
            pipe_d_out_inch = UnitConverter.convert_length(pipe_d_out, pipe_unit, "inch")
            st.caption(f"Inlet/Outlet pipe ratio: {pipe_d_in_inch/pipe_d_out_inch:.2f} (in/out)")
        else:
            pipe_d_in = scenario_data["pipe_d_in"]
            pipe_d_out = scenario_data["pipe_d_out"]
            pipe_unit = scenario_data["pipe_unit"]
            pipe_d_in_inch = pipe_d_in
            pipe_d_out_inch = pipe_d_out
            st.caption("Using valve size for both inlet and outlet pipes")
    
    # Convert all inputs to standard internal units
    flow_std = UnitConverter.convert_flow_rate(flow_value, flow_unit, 
                                               "std m³/h" if fluid_type == "gas" else "m³/h", 
                                               fluid_type, current_temp, current_p1)
    
    # For steam, convert to kg/h
    if fluid_type == "steam":
        flow_std = UnitConverter.convert_flow_rate(flow_value, flow_unit, "kg/h", "steam")
    
    p1_bar = UnitConverter.convert_pressure(p1_value, p1_unit, "bar")
    p2_bar = UnitConverter.convert_pressure(p2_value, p2_unit, "bar")
    temp_c = UnitConverter.convert_temperature(temp_value, temp_unit, "°C")
    
    # Convert viscosity if liquid
    if fluid_type == "liquid":
        visc_cst = UnitConverter.convert_viscosity(visc, visc_unit, "cSt")
    else:
        visc_cst = scenario_data["visc"]
    
    # Convert density if steam
    if fluid_type == "steam":
        rho_kg_m3 = UnitConverter.convert_density(rho, density_unit, "kg/m³")
    else:
        rho_kg_m3 = scenario_data["rho"]
    
    # Convert pipe diameters if not using valve size
    if not use_valve_size:
        pipe_d_in_inch = UnitConverter.convert_length(pipe_d_in, pipe_unit, "inch")
        pipe_d_out_inch = UnitConverter.convert_length(pipe_d_out, pipe_unit, "inch")
    else:
        # Will be set later based on selected valve
        pipe_d_in_inch = 0
        pipe_d_out_inch = 0
    
    return {
        "name": scenario_name,
        "fluid_type": fluid_type,
        "flow": flow_std,
        "flow_display": flow_value,
        "flow_unit": flow_unit,
        "p1": p1_bar,
        "p1_display": p1_value,
        "p1_unit": p1_unit,
        "p2": p2_bar,
        "p2_display": p2_value,
        "p2_unit": p2_unit,
        "temp": temp_c,
        "temp_display": temp_value,
        "temp_unit": temp_unit,
        "sg": sg if fluid_type in ["liquid", "gas"] else scenario_data["sg"],
        "visc": visc_cst if fluid_type == "liquid" else scenario_data["visc"],
        "visc_display": visc if fluid_type == "liquid" else scenario_data["visc"],
        "visc_unit": visc_unit if fluid_type == "liquid" else scenario_data["visc_unit"],
        "pv": pv if fluid_type == "liquid" else scenario_data["pv"],
        "pc": pc if fluid_type == "liquid" else scenario_data["pc"],
        "k": k if fluid_type in ["gas", "steam"] else scenario_data["k"],
        "z": z if fluid_type == "gas" else scenario_data["z"],
        "rho": rho_kg_m3 if fluid_type == "steam" else scenario_data["rho"],
        "rho_display": rho if fluid_type == "steam" else scenario_data["rho"],
        "density_unit": density_unit if fluid_type == "steam" else scenario_data["density_unit"],
        "pipe_d_in": pipe_d_in_inch,
        "pipe_d_in_display": pipe_d_in if not use_valve_size else scenario_data["pipe_d_in"],
        "pipe_d_out": pipe_d_out_inch,
        "pipe_d_out_display": pipe_d_out if not use_valve_size else scenario_data["pipe_d_out"],
        "pipe_unit": pipe_unit,
        "use_valve_size": use_valve_size,
        "fluid_library": fluid_library
    }

def plot_cv_curve(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    # Get valve's Cv characteristics
    openings = sorted(valve.cv_table.keys())
    cv_values = np.array([valve.get_cv_at_opening(op) for op in openings])
    kv_values = np.array([cv_to_kv(cv) for cv in cv_values])
    
    # Create cubic spline interpolation for smooth curve
    cs_cv = CubicSpline(openings, cv_values)
    cs_kv = CubicSpline(openings, kv_values)
    x_smooth = np.linspace(0, 100, 300)
    cv_smooth = cs_cv(x_smooth)
    kv_smooth = cs_kv(x_smooth)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_smooth, 
        y=cv_smooth, 
        mode='lines',
        name='Valve Cv',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=x_smooth, 
        y=kv_smooth, 
        mode='lines',
        name='Valve Kv',
        line=dict(color='green', width=3, dash='dash')
    ))
    
    # Add actual CV points from valve table
    fig.add_trace(go.Scatter(
        x=openings,
        y=cv_values,
        mode='markers',
        name='Actual Cv Points',
        marker=dict(size=8, color='black', symbol='x'),
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=openings,
        y=kv_values,
        mode='markers',
        name='Actual Kv Points',
        marker=dict(size=6, color='darkgreen', symbol='circle'),
        showlegend=True
    ))
    
    for i, (op, req_cv, theoretical_cv) in enumerate(zip(op_points, req_cvs, theoretical_cvs)):
        actual_cv = valve.get_cv_at_opening(op)
        actual_kv = cv_to_kv(actual_cv)
        req_kv = cv_to_kv(req_cv)
        theoretical_kv = cv_to_kv(theoretical_cv)
        
        fig.add_trace(go.Scatter(
            x=[op], 
            y=[actual_cv], 
            mode='markers+text',
            name=f'Scenario {i+1} Operating Point (Cv)',
            marker=dict(size=12, color='red'),
            text=[f'S{i+1} Cv'],
            textposition="top center",
            showlegend=(i==0)
        ))
        fig.add_trace(go.Scatter(
            x=[op], 
            y=[actual_kv], 
            mode='markers+text',
            name=f'Scenario {i+1} Operating Point (Kv)',
            marker=dict(size=10, color='orange'),
            text=[f'S{i+1} Kv'],
            textposition="bottom center",
            showlegend=(i==0)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[req_cv, req_cv],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name=f'Corrected Cv S{i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[req_kv, req_kv],
            mode='lines',
            line=dict(color='orange', dash='dash', width=1),
            name=f'Corrected Kv S{i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[theoretical_cv, theoretical_cv],
            mode='lines',
            line=dict(color='darkred', dash='dot', width=1),
            name=f'Theoretical Cv S{i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[theoretical_kv, theoretical_kv],
            mode='lines',
            line=dict(color='darkorange', dash='dot', width=1),
            name=f'Theoretical Kv S{i+1}',
            showlegend=False
        ))
    
    for i, (req_cv, theoretical_cv) in enumerate(zip(req_cvs, theoretical_cvs)):
        req_kv = cv_to_kv(req_cv)
        theoretical_kv = cv_to_kv(theoretical_cv)
        
        fig.add_annotation(
            x=100,
            y=req_cv,
            text=f'Corrected S{i+1} Cv: {req_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=10,
            align='right',
            font=dict(color='red')
        )
        fig.add_annotation(
            x=100,
            y=req_kv,
            text=f'Corrected S{i+1} Kv: {req_kv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=-10,
            align='right',
            font=dict(color='orange')
        )
        fig.add_annotation(
            x=100,
            y=theoretical_cv,
            text=f'Theoretical S{i+1} Cv: {theoretical_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=30,
            align='right',
            font=dict(color='darkred')
        )
        fig.add_annotation(
            x=100,
            y=theoretical_kv,
            text=f'Theoretical S{i+1} Kv: {theoretical_kv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=-30,
            align='right',
            font=dict(color='darkorange')
        )
    
    fig.update_layout(
        title=f'{valve.size}" Valve Cv & Kv Characteristic',
        xaxis_title='Opening Percentage (%)',
        yaxis_title='Cv/Kv Value',
        legend_title='Legend',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, max(cv_values) * 1.1])
    return fig

def valve_3d_viewer(valve_name, model_url):
    html_code = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer src="{model_url}"
                  alt="{valve_name}"
                  auto-rotate
                  camera-controls
                  autoplay
                  animation-name="valve_animation"
                  style="width: 100%; height: 1000px;">
    </model-viewer>
    """
    components.html(html_code, height=1000)

def main():
    # Global değişkeni kullanacağımızı belirtiyoruz
    global VALVE_DATABASE
    st.set_page_config(
        page_title="Control Valve Sizing",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        html {
            font-size: 18px;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 15px 25px;
            border-radius: 10px 10px 0 0;
            font-size: 18px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 18px;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 18px;
        }
        .warning-card {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .success-card {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .danger-card {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .insufficient-card {
            background-color: #f8d7da;
            border-left: 5px solid #8b0000;
        }
        .cavitation-card {
            background-color: #ffe8cc;
            border-left: 5px solid #fd7e14;
        }
        .velocity-card {
            background-color: #ffd8d8;
            border-left: 5px solid #ff4b4b;
        }
        .calculation-card {
            background-color: #e8f4fd;
            border-left: 5px solid #007bff;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            padding: 10px 0;
        }
        .simulation-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow: auto;
        }
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
        }
        .stMetric {
            font-size: 20px !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .valve-table {
            width: 100%;
            border-collapse: collapse;
        }
        .valve-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .valve-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .status-green {
            background-color: #d4edda;
        }
        .status-yellow {
            background-color: #fff3cd;
        }
        .status-orange {
            background-color: #ffe8cc;
        }
        .status-red {
            background-color: #f8d7da;
        }
        .status-insufficient {
            background-color: #f8d7da;
            border: 2px solid #8b0000;
        }
        .status-velocity {
            background-color: #ffd8d8;
            border: 2px solid #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'valve_database' not in st.session_state:
        st.session_state.valve_database = VALVE_DATABASE
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'valve' not in st.session_state:
        st.session_state.valve = None
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = None
    if 'logo_bytes' not in st.session_state:
        st.session_state.logo_bytes = None
    if 'logo_type' not in st.session_state:
        st.session_state.logo_type = None
    if 'show_simulation' not in st.session_state:
        st.session_state.show_simulation = False
    if 'show_3d_viewer' not in st.session_state:
        st.session_state.show_3d_viewer = False
    if 'recommended_valve' not in st.session_state:
        st.session_state.recommended_valve = None
    if 'all_valve_results' not in st.session_state:
        st.session_state.all_valve_results = None
    if 'calculation_records' not in st.session_state:
        st.session_state.calculation_records = None
    
    # Initialize session state for fluid properties
    for i in range(1, 4):
        if f'sg_{i}' not in st.session_state:
            st.session_state[f'sg_{i}'] = 1.0
        if f'visc_{i}' not in st.session_state:
            st.session_state[f'visc_{i}'] = 1.0
        if f'pv_{i}' not in st.session_state:
            st.session_state[f'pv_{i}'] = 0.023
        if f'pc_{i}' not in st.session_state:
            st.session_state[f'pc_{i}'] = 220.55
        if f'k_{i}' not in st.session_state:
            st.session_state[f'k_{i}'] = 1.4
        if f'z_{i}' not in st.session_state:
            st.session_state[f'z_{i}'] = 1.0
        if f'rho_{i}' not in st.session_state:
            st.session_state[f'rho_{i}'] = 1.0
    
    # Sadece başlık ve alt başlık kalacak - logo kaldırıldı
    st.title("Control Valve Sizing Program")
    st.markdown("**ISA/IEC Standards Compliant Valve Sizing with Enhanced Visualization**")
    st.markdown("VASTAŞ R&D ")
    
    with st.sidebar:
        st.header("VASTAŞ")
        
        if st.session_state.logo_bytes:
            st.image(Image.open(BytesIO(st.session_state.logo_bytes)), use_container_width=True)
        elif os.path.exists("logo.png"):
            st.image(Image.open("logo.png"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x100?text=VASTAŞ+Logo", use_container_width=True)
        
        st.header("Valve Selection")
        valve_options = create_valve_dropdown()
        selected_valve_name = st.selectbox("Select Valve", list(valve_options.keys()))
        selected_valve = valve_options[selected_valve_name]
        
        st.header("Actions")
        calculate_btn = st.button("Calculate Opening", type="primary", use_container_width=True)
        export_btn = st.button("Export PDF Report", use_container_width=True)
        view_3d_btn = st.button("View 3D Model", use_container_width=True)
        show_simulation_btn = st.button("Show Simulation Results", use_container_width=True)
        
        st.header("Valve Details")
        st.markdown(f"**Size:** {selected_valve.size}\"")
        st.markdown(f"**Type:** {'Globe' if selected_valve.valve_type == 3 else 'Axial'}")
        st.markdown(f"**Rating Class:** {selected_valve.rating_class}")
        
        # NOT: Vana notunu göster
        if hasattr(selected_valve, 'note') and selected_valve.note:
            st.markdown(f"**Note:** {selected_valve.note}")
        
        st.markdown(f"**Fl (Liquid Recovery):** {selected_valve.get_fl_at_opening(100):.3f}")
        st.markdown(f"**Xt (Pressure Drop Ratio):** {selected_valve.get_xt_at_opening(100):.3f}")
        st.markdown(f"**Fd (Style Modifier):** {selected_valve.fd:.2f}")
        st.markdown(f"**Internal Diameter:** {selected_valve.diameter:.2f} in")
        
        # Cv Characteristics with Kv column
        st.subheader("Cv & Kv Characteristics")
        cv_data = {
            "Opening %": list(selected_valve.cv_table.keys()), 
            "Cv": list(selected_valve.cv_table.values()),
            "Kv": [cv_to_kv(cv) for cv in selected_valve.cv_table.values()]
        }
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, hide_index=True, height=300)
        
        st.subheader("Fl Characteristics")
        fl_data = {"Opening %": list(selected_valve.fl_table.keys()), "Fl": list(selected_valve.fl_table.values())}
        fl_df = pd.DataFrame(fl_data)
        st.dataframe(fl_df, hide_index=True, height=300)
        
        st.subheader("Xt Characteristics")
        xt_data = {"Opening %": list(selected_valve.xt_table.keys()), "Xt": list(selected_valve.xt_table.values())}
        xt_df = pd.DataFrame(xt_data)
        st.dataframe(xt_df, hide_index=True, height=300)
    
    if view_3d_btn:
        st.session_state.show_3d_viewer = True
        st.session_state.show_simulation = False
    if show_simulation_btn:
        st.session_state.show_simulation = True
        st.session_state.show_3d_viewer = False
    
    with st.sidebar:
        st.header("Valve Management")
        
        if st.button("Reload Valve Database"):
            VALVE_DATABASE = load_valves_from_excel()
            st.session_state.valve_database = VALVE_DATABASE
            st.success("Valve database reloaded!")
        
        with st.expander("Add New Valve"):
            size = st.number_input("Size (inch)", min_value=0.5, step=0.5)
            rating_class = st.selectbox("Rating Class", [150, 300, 600, 900, 1500])
            valve_type = st.selectbox("Valve Type", [3, 4], format_func=lambda x: "Globe" if x == 3 else "Axial")
            fd = st.number_input("Fd", value=1.0)
            diameter = st.number_input("Diameter (inch)", min_value=0.1)
            
            # NOT: Yeni alan eklendi
            note = st.text_input("Note (optional)", placeholder="Örn: Özel uygulama, modifiye vana vb.")
            
            st.subheader("Valve Characteristics Tables")
            
            # Default values for each characteristic
            default_openings = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            default_cv = [0.0, 3.28, 7.39, 12.0, 14.2, 14.9, 15.3, 15.7, 16.0, 16.4, 16.8]
            default_fl = [0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68]
            default_xt = [0.581, 0.605, 0.617, 0.644, 0.764, 0.790, 0.809, 0.813, 0.795, 0.768, 0.74]
            
            # Create editable dataframes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Cv Table**")
                cv_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Cv": default_cv
                })
                cv_edited = st.data_editor(
                    cv_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"cv_editor_{size}_{rating_class}_{valve_type}"
                )
            
            with col2:
                st.markdown("**Fl Table**")
                fl_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Fl": default_fl
                })
                fl_edited = st.data_editor(
                    fl_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"fl_editor_{size}_{rating_class}_{valve_type}"
                )
            
            with col3:
                st.markdown("**Xt Table**")
                xt_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Xt": default_xt
                })
                xt_edited = st.data_editor(
                    xt_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"xt_editor_{size}_{rating_class}_{valve_type}"
                )
            
            if st.button("Add Valve to Database"):
                try:
                    # Convert edited data to dictionaries
                    cv_dict = {}
                    for _, row in cv_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        cv = float(row['Cv'])
                        cv_dict[opening] = cv
                    
                    fl_dict = {}
                    for _, row in fl_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        fl = float(row['Fl'])
                        fl_dict[opening] = fl
                    
                    xt_dict = {}
                    for _, row in xt_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        xt = float(row['Xt'])
                        xt_dict[opening] = xt
                    
                    if note and note.strip() and note.strip().lower() not in ['nan', 'none', 'null']:
                        new_valve = Valve(
                            size_inch=size,
                            rating_class=rating_class,
                            cv_table=cv_dict,
                            fl_table=fl_dict,
                            xt_table=xt_dict,
                            fd=fd,
                            d_inch=diameter,
                            valve_type=valve_type,
                            note=note.strip()
                        )
                    else:
                        new_valve = Valve(
                            size_inch=size,
                            rating_class=rating_class,
                            cv_table=cv_dict,
                            fl_table=fl_dict,
                            xt_table=xt_dict,
                            fd=fd,
                            d_inch=diameter,
                            valve_type=valve_type
                        )
                    add_valve_to_database(new_valve)
                    VALVE_DATABASE = load_valves_from_excel()
                    st.session_state.valve_database = VALVE_DATABASE
                    st.success("Valve added to database!")
                except Exception as e:
                    st.error(f"Error adding valve: {str(e)}")
        
        with st.expander("Delete Valve"):
            valve_options = {get_valve_display_name(v): v for v in VALVE_DATABASE}
            valve_to_delete = st.selectbox("Select Valve to Delete", list(valve_options.keys()))
            
            if st.button("Delete Valve"):
                valve = valve_options[valve_to_delete]
                delete_valve_from_database(valve.size, valve.rating_class, valve.valve_type)
                VALVE_DATABASE = load_valves_from_excel()
                st.session_state.valve_database = VALVE_DATABASE
                st.success("Valve deleted from database!")
    
    tab1, tab2, tab3, tab_results = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3", "Results"])
    
    with tab1:
        scenario1 = scenario_input_form(1)
    with tab2:
        scenario2 = scenario_input_form(2)
    with tab3:
        scenario3 = scenario_input_form(3)
    
    scenarios = []
    if scenario1["flow"] > 0:
        scenarios.append(scenario1)
    if scenario2["flow"] > 0:
        scenarios.append(scenario2)
    if scenario3["flow"] > 0:
        scenarios.append(scenario3)
    st.session_state.scenarios = scenarios
    
    if calculate_btn:
        if not scenarios:
            st.error("Please define at least one scenario with flow > 0.")
            st.stop()
        try:
            selected_valve_results = []
            calculation_records = []
            for scenario in scenarios:
                result = evaluate_valve_for_scenario_with_record(selected_valve, scenario)
                selected_valve_results.append(result)
                calculation_records.append(result.get("calculation_record"))
            
            recommended_valve, all_valve_results = find_recommended_valve(scenarios)
            st.session_state.results = {
                "selected_valve": selected_valve,
                "selected_valve_results": selected_valve_results,
                "recommended_valve": recommended_valve,
                "all_valve_results": all_valve_results,
                "calculation_records": calculation_records
            }
            st.success("Calculation completed with comprehensive records!")
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            st.error(traceback.format_exc())
    
    with tab_results:
        if st.session_state.results:
            results = st.session_state.results
            selected_valve = results["selected_valve"]
            selected_valve_results = results["selected_valve_results"]
            recommended_valve = results["recommended_valve"]
            calculation_records = results.get("calculation_records", [])
            
            if recommended_valve:
                st.subheader("Recommended Valve")
                st.markdown(f"**{recommended_valve['display_name']}** - Score: {recommended_valve['score']:.1f}")
                
                # Show each scenario result for recommended valve
                for i, scenario in enumerate(scenarios):
                    result = recommended_valve["results"][i]
                    actual_cv = recommended_valve["valve"].get_cv_at_opening(result["op_point"])
                    actual_kv = cv_to_kv(actual_cv)
                    
                    # Determine status class
                    status_class = ""
                    if "Insufficient" in result["warning"]:
                        status_class = "insufficient-card"
                    elif "High velocity" in result["warning"]:
                        status_class = "velocity-card"
                    elif result["status"] == "green":
                        status_class = "success-card"
                    elif result["status"] == "yellow":
                        status_class = "warning-card"
                    elif result["status"] == "orange":
                        status_class = "cavitation-card"
                    elif result["status"] == "red":
                        status_class = "danger-card"
                    
                    # Combine warnings
                    warn_msgs = []
                    if result["warning"]:
                        warn_msgs.append(result["warning"])
                    if result["cavitation_info"]:
                        warn_msgs.append(result["cavitation_info"])
                    warn_text = ", ".join(warn_msgs)
                    
                    with st.container():
                        st.markdown(f"<div class='result-card {status_class}'>", unsafe_allow_html=True)
                        cols = st.columns([1.8, 1.5, 1.5, 1.5, 1, 1, 1, 1.5])
                        cols[0].markdown(f"**{scenario['name']}**")
                        cols[1].metric("Req Cv/Kv", f"{result['req_cv']:.1f}/{result['req_kv']:.1f}")
                        cols[2].metric("Theo Cv/Kv", f"{result['theoretical_cv']:.1f}/{result['theoretical_kv']:.1f}")
                        cols[3].metric("Valve Cv/Kv", f"{actual_cv:.1f}/{actual_kv:.1f}")
                        cols[4].metric("Valve Size", f"{recommended_valve['valve'].size}\"")
                        cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                        cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                      delta_color="inverse" if result['margin'] < 0 else "normal")
                        cols[7].markdown(f"**{warn_text}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("Why this valve is recommended"):
                    st.markdown("""
                    - **Green status**: Optimal performance (good operating range, no cavitation)
                    - **Yellow status**: Acceptable but suboptimal (moderate cavitation or bad opening)
                    - **Orange status**: Severe cavitation risk
                    - **Red status**: Choked flow (unacceptable)
                    - **Dark red**: Insufficient capacity (valve undersized)
                    - **Pink**: High velocity warning
                    """)
                    st.markdown(f"""
                    **Selection criteria**:
                    - Highest overall score considering operating point, cavitation risk, and flow conditions
                    - Valve size: {recommended_valve['valve'].size}\"
                    - Valve type: {'Globe' if recommended_valve['valve'].valve_type == 3 else 'Axial'}
                    """)
            else:
                st.warning("No suitable valve found for all scenarios. Consider modifying your scenarios.")
            
            st.subheader(f"Selected Valve: {get_valve_display_name(selected_valve)} Cv & Kv Characteristic")
            fig = plot_cv_curve(
                selected_valve, 
                [r["op_point"] for r in selected_valve_results],
                [r["req_cv"] for r in selected_valve_results],
                [r["theoretical_cv"] for r in selected_valve_results],
                [s["name"] for s in scenarios]
            )
            st.plotly_chart(fig, use_container_width=True, key="main_cv_curve")
            
            st.subheader("Selected Valve Performance")
            for i, scenario in enumerate(scenarios):
                result = selected_valve_results[i]
                actual_cv = selected_valve.get_cv_at_opening(result["op_point"])
                actual_kv = cv_to_kv(actual_cv)
                status = "success-card"
                if "Insufficient" in result["warning"]:
                    status = "insufficient-card"
                elif "High velocity" in result["warning"]:
                    status = "velocity-card"
                elif result["status"] == "yellow":
                    status = "warning-card"
                elif result["status"] == "orange":
                    status = "cavitation-card"
                elif result["status"] == "red":
                    status = "danger-card"
                warn_msgs = []
                if result["warning"]:
                    warn_msgs.append(result["warning"])
                if result["cavitation_info"]:
                    warn_msgs.append(result["cavitation_info"])
                warn_text = ", ".join(warn_msgs)
                
                with st.container():
                    st.markdown(f"<div class='result-card {status}'>", unsafe_allow_html=True)
                    cols = st.columns([1.8, 1.5, 1.5, 1.5, 1, 1, 1, 1.5])
                    cols[0].markdown(f"**{scenario['name']}**")
                    cols[1].metric("Req Cv/Kv", f"{result['req_cv']:.1f}/{result['req_kv']:.1f}")
                    cols[2].metric("Theo Cv/Kv", f"{result['theoretical_cv']:.1f}/{result['theoretical_kv']:.1f}")
                    cols[3].metric("Valve Cv/Kv", f"{actual_cv:.1f}/{actual_kv:.1f}")
                    cols[4].metric("Valve Size", f"{selected_valve.size}\"")
                    cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                    cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                  delta_color="inverse" if result['margin'] < 0 else "normal")
                    cols[7].markdown(f"**{warn_text}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    with st.expander(f"Detailed Calculations for {scenario['name']}", expanded=False):
                        st.subheader("Calculation Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Valve Diameter:** {selected_valve.diameter} inch")
                            
                            if scenario["fluid_type"] == "liquid":
                                fl_eff = result.get('fl_effective', result['details'].get('fl_at_op', 'N/A'))
                                if isinstance(fl_eff, (int, float)):
                                    st.markdown(f"**Fl (Liquid Recovery):** {fl_eff:.3f}")
                                    if result.get('fl_effective') and not scenario.get("use_valve_size", True):
                                        st.markdown(f"*Piping corrected (FLP) value*")
                                else:
                                    st.markdown(f"**Fl (Liquid Recovery):** {fl_eff}")
                                    
                            elif scenario["fluid_type"] in ["gas", "steam"]:
                                xt_eff = result.get('xt_effective', result['details'].get('xt_at_op', 'N/A'))
                                if isinstance(xt_eff, (int, float)):
                                    st.markdown(f"**xT/xTP (Pressure Drop Ratio):** {xt_eff:.4f}")
                                    if result.get('xt_effective') and not scenario.get("use_valve_size", True):
                                        st.markdown(f"*Piping corrected (xTP) value*")
                                else:
                                    st.markdown(f"**xT/xTP (Pressure Drop Ratio):** {xt_eff}")
                            
                            st.markdown(f"**Fd (Valve Style Modifier):** {selected_valve.fd:.2f}")
                            
                            fp_val = result.get('fp', result['details'].get('fp', 1.0))
                            if isinstance(fp_val, (int, float)):
                                st.markdown(f"**Fp (Piping Factor):** {fp_val:.4f}")
                                cv_used = result['details'].get('cv_used_for_fp', 'N/A')
                                if cv_used != 'N/A':
                                    st.markdown(f"*Calculated using Cv at operating point: {cv_used:.1f}*")
                            else:
                                st.markdown(f"**Fp (Piping Factor):** {fp_val}")
                            
                        with col2:
                            if scenario["fluid_type"] == "liquid":
                                ff_val = result['details'].get('ff', 0.96)
                                if isinstance(ff_val, (int, float)):
                                    st.markdown(f"**FF (Critical Pressure Ratio):** {ff_val:.4f}")
                                else:
                                    st.markdown(f"**FF (Critical Pressure Ratio):** {ff_val}")
                                    
                                fr_val = result['details'].get('fr', 1.0)
                                if isinstance(fr_val, (int, float)):
                                    st.markdown(f"**Fr (Viscosity Correction):** {fr_val:.4f}")
                                else:
                                    st.markdown(f"**Fr (Viscosity Correction):** {fr_val}")
                                
                                reynolds_val = result['details'].get('reynolds', 0)
                                if isinstance(reynolds_val, (int, float)):
                                    st.markdown(f"**Reynolds Number:** {reynolds_val:.0f}")
                                else:
                                    st.markdown(f"**Reynolds Number:** {reynolds_val}")
                            
                        dp_max_val = result['details'].get('dp_max', 0)
                        if isinstance(dp_max_val, (int, float)):
                            st.markdown(f"**Max Pressure Drop (ΔPmax):** {dp_max_val:.2f} bar")
                        else:
                            st.markdown(f"**Max Pressure Drop (ΔPmax):** {dp_max_val} bar")
                        
                        # Show corrected velocity calculations
                        st.subheader("Flow Velocities with Pressure-Dependent Volumetric Flow")
                        velocity_details = result.get('velocity_details', {})
                        
                        col_vel1, col_vel2, col_vel3 = st.columns(3)
                        with col_vel1:
                            st.markdown("**Inlet Conditions:**")
                            st.markdown(f"Flow: {velocity_details.get('inlet_flow_m3h', 0):.2f} m³/h")
                            st.markdown(f"Density: {velocity_details.get('inlet_density', 0):.2f} kg/m³")
                            st.markdown(f"Velocity: {result.get('inlet_velocity', 0):.2f} m/s")
                            st.markdown(f"ΔP (velocity): {result.get('inlet_pressure_drop', 0):.4f} bar")
                        
                        with col_vel2:
                            st.markdown("**Outlet Conditions:**")
                            st.markdown(f"Flow: {velocity_details.get('outlet_flow_m3h', 0):.2f} m³/h")
                            st.markdown(f"Density: {velocity_details.get('outlet_density', 0):.2f} kg/m³")
                            st.markdown(f"Velocity: {result.get('outlet_velocity', 0):.2f} m/s")
                            st.markdown(f"ΔP (velocity): {result.get('outlet_pressure_drop', 0):.4f} bar")
                        
                        with col_vel3:
                            st.markdown("**Orifice Conditions:**")
                            st.markdown(f"Flow: {velocity_details.get('orifice_flow_m3h', 0):.2f} m³/h")
                            st.markdown(f"Density: {velocity_details.get('orifice_density', 0):.2f} kg/m³")
                            st.markdown(f"Velocity: {result.get('orifice_velocity', 0):.2f} m/s")
                            st.markdown(f"ΔP (velocity): {result.get('orifice_pressure_drop', 0):.4f} bar")
                        
                        # Show flow ratios for compressible fluids
                        if scenario["fluid_type"] in ["gas", "steam"]:
                            st.markdown(f"**Flow Ratio (outlet/inlet):** {velocity_details.get('flow_ratio', 0):.3f}")
                            st.markdown(f"**Density Ratio (outlet/inlet):** {velocity_details.get('density_ratio', 0):.3f}")
                            st.markdown(f"**Pressure Ratio (outlet/inlet):** {velocity_details.get('pressure_ratio', 0):.3f}")
                            st.info(f"Note: For {scenario['fluid_type']}, volumetric flow changes with pressure. Outlet flow is {velocity_details.get('flow_ratio', 0):.3f}× inlet flow.")
                        
                        # Velocity warning if present
                        if "High velocity" in result["warning"]:
                            st.warning(f"**Velocity Warning:** {result['warning']}")
                        
                        # Add Fp Calculation Details section
                        if result.get('fp_details'):
                            st.subheader("Fp (Piping Geometry Factor) Calculation Details")
                            fp_details = result['fp_details']
                            
                            col_fp1, col_fp2 = st.columns(2)
                            with col_fp1:
                                st.markdown(f"**Valve Diameter:** {fp_details.get('valve_d_inch', 'N/A')} inch")
                                st.markdown(f"**Inlet Pipe Diameter:** {fp_details.get('pipe_d_in_inch', 'N/A')} inch")
                                st.markdown(f"**Outlet Pipe Diameter:** {fp_details.get('pipe_d_out_inch', 'N/A')} inch")
                                st.markdown(f"**Cv at operating point:** {fp_details.get('cv_op', 'N/A'):.1f}")
                                st.markdown(f"**d_ratio_in (valve/inlet):** {fp_details.get('d_ratio_in', 'N/A'):.4f}")
                                st.markdown(f"**d_ratio_out (valve/outlet):** {fp_details.get('d_ratio_out', 'N/A'):.4f}")
                            
                            with col_fp2:
                                st.markdown(f"**K1 (inlet reducer):** {fp_details.get('K1', 'N/A'):.4f}")
                                st.markdown(f"**K2 (outlet reducer):** {fp_details.get('K2', 'N/A'):.4f}")
                                st.markdown(f"**KB1 (inlet Bernoulli):** {fp_details.get('KB1', 'N/A'):.4f}")
                                st.markdown(f"**KB2 (outlet Bernoulli):** {fp_details.get('KB2', 'N/A'):.4f}")
                                st.markdown(f"**ΣK = K1 + K2 + KB1 - KB2:** {fp_details.get('sumK', 'N/A'):.4f}")
                                st.markdown(f"**N2 constant:** {fp_details.get('N2', 'N/A')}")
                            
                            st.markdown(f"**Term = 1 + (ΣK/N2) * (Cv/d²)²:** {fp_details.get('term', 'N/A'):.4f}")
                            st.markdown(f"**Fp = 1 / √(Term):** {fp_details.get('Fp', 'N/A'):.4f}")
                            
                            if 'formula' in fp_details:
                                st.markdown(f"**Formula:** {fp_details['formula']}")
                            
                            with st.expander("Formulas Used"):
                                st.markdown(f"**ΣK formula:** {fp_details.get('components', 'N/A')}")
                                st.markdown(f"**K1 formula:** {fp_details.get('K1_formula', 'N/A')}")
                                st.markdown(f"**K2 formula:** {fp_details.get('K2_formula', 'N/A')}")
                                st.markdown(f"**KB1 formula:** {fp_details.get('KB1_formula', 'N/A')}")
                                st.markdown(f"**KB2 formula:** {fp_details.get('KB2_formula', 'N/A')}")
                        
                        if scenario["fluid_type"] == "liquid":
                            if result["details"].get('cavitation_severity'):
                                st.subheader("Cavitation Analysis")
                                st.markdown(f"**Status:** {result['details']['cavitation_severity']}")
                                
                                sigma_val = result['details'].get('sigma', 0)
                                if isinstance(sigma_val, (int, float)):
                                    st.markdown(f"**Sigma (σ):** {sigma_val:.2f}")
                                else:
                                    st.markdown(f"**Sigma (σ):** {sigma_val}")
                                
                                km_val = result['details'].get('km', 0)
                                if isinstance(km_val, (int, float)):
                                    st.markdown(f"**Km (Valve Recovery Coefficient):** {km_val:.2f}")
                                else:
                                    st.markdown(f"**Km (Valve Recovery Coefficient):** {km_val}")
                        
                        if scenario["fluid_type"] in ["gas", "steam"]:
                            cv_alternative = result['details'].get('cv_alternative', 0)
                            if isinstance(cv_alternative, (int, float)) and cv_alternative > 0:
                                st.markdown(f"**Alternative Cv (x_actual, Y=0.667):** {cv_alternative:.1f}")
                                st.markdown(f"**Alternative Kv (x_actual, Y=0.667):** {cv_to_kv(cv_alternative):.1f}")
                                
                                theoretical_cv = result['theoretical_cv']
                                if theoretical_cv > 0:
                                    diff_percent = ((cv_alternative - theoretical_cv) / theoretical_cv) * 100
                                    st.markdown(f"**Difference from theoretical Cv:** {diff_percent:+.1f}%")
                                
                                st.markdown(f"**Method:** {result['details'].get('alternative_method', 'N/A')}")
                                
                                with st.expander("Explanation of Alternative Calculation"):
                                    st.markdown("""
                                    **Alternative Cv Calculation Method:**
                                    - Uses **x_actual** (actual pressure drop ratio) instead of x_crit
                                    - Uses **constant Y=0.667** (choked flow expansion factor)
                                    - This shows what the Cv would be if we treated the flow as choked regardless of the actual x_crit value
                                    
                                    **When to use this comparison:**
                                    - When x_actual is close to x_crit but slightly below
                                    - To understand the sensitivity of Cv to expansion factor assumptions
                                    - For conservative sizing approaches
                                    """)

                            st.subheader("Choked Flow Analysis")
                            st.markdown(f"**Status:** {result['cavitation_info']}")
                            
                            x_actual_val = result['details'].get('x_actual', 0)
                            if isinstance(x_actual_val, (int, float)):
                                st.markdown(f"**Pressure Drop Ratio (x):** {x_actual_val:.4f}")
                            else:
                                st.markdown(f"**Pressure Drop Ratio (x):** {x_actual_val}")
                            
                            x_crit_val = result['details'].get('x_crit', 0)
                            if isinstance(x_crit_val, (int, float)):
                                st.markdown(f"**Critical Pressure Drop Ratio (x_crit):** {x_crit_val:.4f}")
                            else:
                                st.markdown(f"**Critical Pressure Drop Ratio (x_crit):** {x_crit_val}")
                            
                            xt_at_op_val = result['details'].get('xt_at_op', 0)
                            xt_op_point = result['details'].get('xt_op_point', 'N/A')
                            if isinstance(xt_at_op_val, (int, float)):
                                st.markdown(f"**Pressure Drop Ratio Factor (xT or xTP):** {xt_at_op_val:.4f}")
                                if xt_op_point != 'N/A':
                                    st.markdown(f"*Calculated at {xt_op_point}% opening*")
                            else:
                                st.markdown(f"**Pressure Drop Ratio Factor (xT or xTP):** {xt_at_op_val}")
                            
                            if isinstance(x_crit_val, (int, float)) and isinstance(scenario['p1'], (int, float)):
                                choked_dp = x_crit_val * scenario['p1']
                                st.markdown(f"**Choked Pressure Drop:** {choked_dp:.2f} bar")
                            else:
                                st.markdown("**Choked Pressure Drop:** N/A")
                        
                        st.subheader("Flow Rate vs Pressure Drop")
                        flow_fig = generate_flow_vs_dp_graph(
                            scenario,
                            selected_valve,
                            result["op_point"],
                            result["details"],
                            result["req_cv"]
                        )
                        st.plotly_chart(flow_fig, use_container_width=True, key=f"flow_dp_{i}")
                        
                        # Show comprehensive calculation record
                        if calculation_records and i < len(calculation_records) and calculation_records[i]:
                            with st.expander("View Complete Calculation Record", expanded=False):
                                record = calculation_records[i]
                                
                                # Create tabs for different sections of the calculation record
                                tab_calc, tab_iter, tab_formulas, tab_download = st.tabs([
                                    "Calculation Steps", 
                                    "Iterations", 
                                    "Formulas",
                                    "Download"
                                ])
                                
                                with tab_calc:
                                    st.markdown("### Step-by-Step Calculation")
                                    steps = record.get_formatted_steps()
                                    for step in steps:
                                        st.markdown(step)
                                
                                with tab_iter:
                                    st.markdown("### Iteration History")
                                    if record.iterations:
                                        iterations = record.get_formatted_iterations()
                                        for iteration in iterations:
                                            st.markdown(iteration)
                                    else:
                                        st.info("No iteration data available")
                                
                                with tab_formulas:
                                    st.markdown("### Formulas Used")
                                    for formula in record.formulas:
                                        with st.expander(f"Formula: {formula['name']}"):
                                            st.code(formula['formula'])
                                            st.markdown(f"**Explanation:** {formula['explanation']}")
                                            if formula.get('variables'):
                                                st.markdown("**Variables:**")
                                                for var_key, var_desc in formula['variables'].items():
                                                    st.markdown(f"- **{var_key}**: {var_desc}")
                                
                                with tab_download:
                                    st.markdown("### Download Calculation Report")
                                    # Create a download button for the detailed calculation record
                                    record_text = record.generate_detailed_report()
                                    st.download_button(
                                        label="Download Complete Calculation Record",
                                        data=record_text,
                                        file_name=f"Complete_Calculation_Record_{scenario['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain"
                                    )
                                    
                                    # Also show summary download
                                    summary = record.get_summary()
                                    summary_text = f"Calculation Summary for {scenario['name']}\n"
                                    summary_text += f"Fluid Type: {scenario['fluid_type']}\n"
                                    summary_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                    
                                    summary_text += "Final Results:\n"
                                    for key, data in summary['results'].items():
                                        summary_text += f"  {key}: {data['value']} {data.get('unit', '')}\n"
                                    
                                    st.download_button(
                                        label="Download Results Summary",
                                        data=summary_text,
                                        file_name=f"Results_Summary_{scenario['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain"
                                    )
                                    
                                    st.markdown("---")
                                    st.markdown("**Report includes:**")
                                    st.markdown("• All input parameters")
                                    st.markdown("• Every calculation step with formulas and actual values")
                                    st.markdown("• Complete iteration history")
                                    st.markdown("• All intermediate values")
                                    st.markdown("• Assumptions and warnings")
                                    st.markdown("• Final results")
                        
                        # Show iteration details if available
                        if result.get('iterations'):
                            with st.expander("Iteration Details Table", expanded=False):
                                st.markdown("**Complete Iteration History (including final iteration):**")
                                iter_df = pd.DataFrame(result['iterations'])
                                
                                display_columns = ['iteration', 'opening', 'cv_at_op', 'kv_at_op', 
                                                 'fl_at_op', 'xt_at_op', 'fp', 'cv_req', 'kv_req', 
                                                 'new_opening', 'convergence_diff', 'note']
                                available_columns = [col for col in display_columns if col in iter_df.columns]
                                st.dataframe(iter_df[available_columns])
                                
                                if result.get('converged'):
                                    st.success(f"✓ Converged after {len(result['iterations'])} iterations")
                                    st.info(f"**Final iteration (#{len(result['iterations'])}) shows the converged values with all corrections applied.**")
                                else:
                                    st.warning(f"⚠ Did not fully converge after {len(result['iterations'])} iterations (using last values)")
            
            st.subheader("All Valves Evaluation")
            st.markdown("""
            **Status colors**:
            - <span style="background-color:#d4edda; padding:2px 5px;">Green</span>: Optimal
            - <span style="background-color:#fff3cd; padding:2px 5px;">Yellow</span>: Warning (moderate issue)
            - <span style="background-color:#ffe8cc; padding:2px 5px;">Orange</span>: Severe cavitation
            - <span style="background-color:#f8d7da; padding:2px 5px;">Red</span>: Choked flow (unacceptable)
            - <span style="background-color:#f8d7da; border:2px solid #8b0000; padding:2px 5px;">Dark Red</span>: Insufficient capacity
            - <span style="background-color:#ffd8d8; padding:2px 5px;">Pink</span>: High velocity
            """, unsafe_allow_html=True)
            all_valves_table_html = """
            <table class="valve-table">
                <thead>
                    <tr>
                        <th>Valve</th>
            """
            for i, scenario in enumerate(scenarios):
                all_valves_table_html += f'<th>{scenario["name"]} Status</th>'
            all_valves_table_html += """
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
            """
            all_valve_results = sorted(
                st.session_state.results["all_valve_results"], 
                key=lambda x: x["score"], 
                reverse=True
            )
            for valve_result in all_valve_results:
                all_valves_table_html += f'<tr><td>{valve_result["display_name"]}</td>'
                for result in valve_result["results"]:
                    status_class = ""
                    if "Insufficient" in result["warning"]:
                        status_class = "status-insufficient"
                    elif "High velocity" in result["warning"]:
                        status_class = "status-velocity"
                    elif result["status"] == "green":
                        status_class = "status-green"
                    elif result["status"] == "yellow":
                        status_class = "status-yellow"
                    elif result["status"] == "orange":
                        status_class = "status-orange"
                    elif result["status"] == "red":
                        status_class = "status-red"
                    all_valves_table_html += f'<td class="{status_class}">{result["status"]}</td>'
                all_valves_table_html += f'<td>{valve_result["score"]:.1f}</td></tr>'
            all_valves_table_html += "</tbody></table>"
            st.markdown(all_valves_table_html, unsafe_allow_html=True)
    
    # Handle export button
    if export_btn:
        if st.session_state.results is None:
            st.error("Please run the calculation first.")
        else:
            with st.spinner("Generating PDF report with calculation details..."):
                # Prepare data for PDF
                scenarios = st.session_state.scenarios
                valve = st.session_state.results["selected_valve"]
                op_points = [r["op_point"] for r in st.session_state.results["selected_valve_results"]]
                req_cvs = [r["req_cv"] for r in st.session_state.results["selected_valve_results"]]
                warnings = [r["warning"] for r in st.session_state.results["selected_valve_results"]]
                cavitation_info = [r["cavitation_info"] for r in st.session_state.results["selected_valve_results"]]
                theoretical_cvs = [r["theoretical_cv"] for r in st.session_state.results["selected_valve_results"]]
                calculation_records = st.session_state.results.get("calculation_records", [])
                
                # Generate the Cv curve plot for PDF
                plot_bytes = plot_cv_curve_matplotlib(valve, op_points, req_cvs, theoretical_cvs, [s["name"] for s in scenarios])
                
                # Generate one Flow vs DP plot (for the first scenario) for PDF
                flow_dp_plot_bytes = None
                if scenarios:
                    flow_dp_plot_bytes = plot_flow_vs_dp_matplotlib(
                        scenarios[0],
                        valve,
                        op_points[0],
                        st.session_state.results["selected_valve_results"][0]["details"],
                        req_cvs[0]
                    )
                
                # Generate PDF with calculation records
                logo_bytes = st.session_state.logo_bytes
                logo_type = st.session_state.logo_type
                pdf_bytes_io = generate_pdf_report(
                    scenarios, valve, op_points, req_cvs, warnings, cavitation_info, 
                    plot_bytes, flow_dp_plot_bytes, logo_bytes, logo_type,
                    calculation_records
                )
                
                # Offer download
                st.success("PDF report with calculation details generated!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes_io,
                    file_name=f"Valve_Sizing_Report_with_Calculations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # Handle 3D viewer and simulation display
    if st.session_state.show_3d_viewer:
        valve_name = get_valve_display_name(selected_valve)
        model_url = VALVE_MODELS.get(valve_name, None)
        if model_url:
            st.subheader(f"3D Model: {valve_name}")
            valve_3d_viewer(valve_name, model_url)
        else:
            st.warning(f"3D model not available for {valve_name}")
        if st.button("Close 3D Viewer"):
            st.session_state.show_3d_viewer = False
    
    if st.session_state.show_simulation:
        valve_name = get_valve_display_name(selected_valve)
        sim_image_url = get_simulation_image(valve_name)
        st.subheader(f"CFD Simulation Results: {valve_name}")
        st.image(sim_image_url, use_container_width=True)
        if st.button("Close Simulation"):
            st.session_state.show_simulation = False

# Run the main function
if __name__ == "__main__":
    main()
