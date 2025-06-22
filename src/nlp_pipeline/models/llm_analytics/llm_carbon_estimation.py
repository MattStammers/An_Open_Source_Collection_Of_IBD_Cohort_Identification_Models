"""
Example Script for LLM Carbon Estimation

This is how the LLM CO2 estimates were calculated as best-proxy measurements.

The other calculations use codecarbon with a standard emission factor as per the below

Author: Matt Stammers / UHSFT
"""
# --------------------------------------------------------------------------- #
# Carbon Calc for LLM Demo Script                                             #
# --------------------------------------------------------------------------- #

# Data
power_watts = 253  # Power in watts
time_minutes = 1075  # Time in minutes
conversion_factor = 0.20705  # CO2e per kWh

# Step 1: Necessary Conversions
power_kw = power_watts / 1000  # Convert watts to kilowatts
time_hours = time_minutes / 60  # Convert minutes to hours

# Step 2: Energy Consumption in kWh
energy_kwh = power_kw * time_hours

# Step 3: Calculate CO2 emissions in Kg
co2_emissions = energy_kwh * conversion_factor

# Output results
print(f"Energy consumed: {energy_kwh:.4f} kWh")
print(f"CO2 emissions: {co2_emissions:.4f} Kg CO2e")
