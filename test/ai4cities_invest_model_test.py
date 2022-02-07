from ai4cities.ai4cities_invest_model import solve_model, model_invest, model_invest_input, model_invest_results
import pandas as pd
import numpy as np
import copy

# Set solver
solver = {'name':"cbc",'path': "C:/cbc-win64/cbc"}

# Import data
data = pd.read_csv('./data/data.csv', header=[0], sep=',', index_col=0, parse_dates = True)


# Copy data to new dataframe
df = copy.copy(data)


# Assign value to every unique month-year
df['TM'] = df.index.month
df['TY'] = df.index.year
df['month_order'] = df['TM'] + (df['TY'] - df['TY'][0])*12 - df['TM'][0]+1
df = df.drop(columns=['TM', 'TY'])

    

#%% Tarrif structures

# Example 1 - Fixed energy tariff
# Prices for Affärsverken Elnät AB (Fuse 16A, appartment)
fixed_charge = 14.5 # €/Month

df['grid_fee_energy_import'] = 45*0.001 # €/kWh
df['grid_fee_energy_export'] = 0 # €/kWh
df['grid_fee_power_import'] = 0 # €/kW-month
df['grid_fee_power_export'] = 0 # €/kW-month




# Example 2 - Power based tariff
# Prices for Bjäre Kraft ek för (Fuse 16, house)
fixed_charge = 14.1 # €/Month

df['grid_fee_energy_import'] = 0 # €/kWh
df['grid_fee_energy_export'] = 0 # €/kWh
df['grid_fee_power_import'] = 0 # €/kW-month
df['grid_fee_power_export'] = 0 # €/kW-month

# Set grid energy fee for different months, days and hours
for i in [1,2,3,11,12]:
    df.loc[(df.index.month == i),'grid_fee_power_import'] = 12.6

for i in [4,5,6,7,8,9,10]:
    for j in list(range(0,5)):
        for k in list(range(9,19)):
            df.loc[(df.index.month == i) & (df.index.weekday == j) & (df.index.hour == k),'grid_fee_power_import'] = 7.5




# Example 3 - Time based tariff
# Prices for Ellevio AB Dalarna Södra (Fuse 16A, house)
fixed_charge = 25.5 # €/Month

df['grid_fee_energy_import'] = 9*0.001 # €/kWh
df['grid_fee_energy_export'] = 0 # €/kWh
df['grid_fee_power_import'] = 0 # €/kW-month
df['grid_fee_power_export'] = 0 # €/kW-month

# Set grid energy fee for different months, days and hours
for i in [1,2,3,11,12]:
    for j in list(range(0,5)):
        for k in list(range(8,22)):
            df.loc[(df.index.month == i) & (df.index.weekday == j) & (df.index.hour == k),'grid_fee_energy_import'] = 58*0.001


for i in [4,5,6,7,8,9,10]:
    df.loc[(df.index.month == i),'grid_fee_energy_import'] = 9*0.001




#%% Create data dictionary and solve model
data_dict = {
        'invest_unit_cost_battery': 20,
        'invest_unit_cost_pv': 30,
        'invest_unit_cost_heat_pump': 10,
        'invest_unit_cost_electric_boiler': 20,
        'invest_unit_cost_fuel_boiler': 10,
        'invest_unit_cost_district_heat': 5,
        'invest_unit_cost_thermal_storage': 5,
        
        'pv_lifespan': 25,
        'battery_lifespan': 10,
        'heat_pump_lifespan': 25,
        'electric_boiler_lifespan': 25,
        'fuel_boiler_lifespan': 25,
        'district_heat_lifespan': 50,
        'thermal_storage_lifespan': 30,
        
        'discount_rate': 0.03,
        'salvage_rate': 0.0,
        'heat_capacity_oversize': 0.1,
        
        'pv_capacity_max': 20,
        'pv_capacity_min': 0.0, 
        
        'battery_capacity_max': 5.0,
        'battery_capacity_min': 0.0,
        'battery_soc_min': 0.0,
        'battery_charge_max': 0.5,
        'battery_discharge_max': 0.5,
        'battery_efficiency_charge': 0.9,
        'battery_efficiency_discharge': 0.9,
        'battery_grid_charging': True,
        'battery_soc_ini': 0.0,
        'battery_soc_fin': 0.0,
        
        'heat_pump_capacity_max': 232,
        'heat_pump_capacity_min': 0.0,
        'heat_pump_cop': 2.5,
        
        'electric_boiler_capacity_max': 0.0,
        'electric_boiler_capacity_min': 0.0,
        'electric_boiler_efficiency': 0.9,
        
        'fuel_boiler_capacity_max': 0.0,
        'fuel_boiler_capacity_min': 0.0,
        'fuel_boiler_efficiency': 0.9,
        
        'district_heat_capacity_max': 0.0,
        'district_heat_capacity_min': 0.0,

        'thermal_storage_capacity_max': 0.0,
        'thermal_storage_capacity_min': 0.0,
        'thermal_storage_charge_max': 1.0,
        'thermal_storage_discharge_max': 1.0,
        'thermal_storage_losses': 0.05,
        'thermal_storage_soc_ini': 0.0,
        'thermal_storage_soc_fin': 0.0,

        'power_demand': df['Electricity demand [kWh]'].to_list(),  
        'power_generation_pu': df['PV 1kWp 180Azim 40Tilt Physical [kW]'].to_list(),
        'heat_demand': df['Heat demand [kWh]'].to_list(),

        'electricity_price_buy': (df['DAM Price [EUR/MWh]']/1000).to_list(),
        'electricity_price_sell': (0.99*df['DAM Price [EUR/MWh]']/1000).to_list(),
        'fuel_price': list(20*np.ones(len(df))),
        'district_heat_price': list(10*np.ones(len(df))),
        
        'grid_fee_fixed': fixed_charge,
        'grid_fee_energy_import': df['grid_fee_energy_import'].to_list(),
        'grid_fee_energy_export': df['grid_fee_energy_export'].to_list(),
        'grid_fee_power_import': df['grid_fee_power_import'].to_list(),
        'grid_fee_power_export': df['grid_fee_power_export'].to_list(),
        
        'specific_emission_grid': list(5*np.ones(len(df))),
        'specific_emission_fuel': list(15*np.ones(len(df))),
        'specific_emission_district_heat': list(10*np.ones(len(df))),
        
        'month_order': df['month_order'].to_list(),
        'dt': 1.0, 
}
    

# Create model data structure
model_data = model_invest_input(data_dict) 

# Create model instance with data
model_instance = model_invest(model_data)

# Solve
solution = solve_model(model_instance, solver) 

# Get results
results = model_invest_results(solution)