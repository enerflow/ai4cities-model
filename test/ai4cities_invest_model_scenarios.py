from ai4cities.ai4cities_invest_model import solve_model, model_invest, model_invest_input, model_invest_results
import pandas as pd
import numpy as np
import copy

# Set solver
#solver = {'name':"cbc",'path': "C:/cbc-win64/cbc"}
solver = {'name':"cbc"}

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

df['grid_energy_import_fee'] = 45*0.001 # €/kWh
df['grid_energy_export_fee'] = 0 # €/kWh
df['grid_power_import_fee'] = 0 # €/kW-month
df['grid_power_export_fee'] = 0 # €/kW-month




# Example 2 - Power based tariff
# Prices for Bjäre Kraft ek för (Fuse 16, house)
fixed_charge = 14.1 # €/Month

df['grid_energy_import_fee'] = 0 # €/kWh
df['grid_energy_export_fee'] = 0 # €/kWh
df['grid_power_import_fee'] = 0 # €/kW-month
df['grid_power_export_fee'] = 0 # €/kW-month

# Set grid energy fee for different months, days and hours
for i in [1,2,3,11,12]:
    df.loc[(df.index.month == i),'grid_power_import_fee'] = 12.6

for i in [4,5,6,7,8,9,10]:
    for j in list(range(0,5)):
        for k in list(range(9,19)):
            df.loc[(df.index.month == i) & (df.index.weekday == j) & (df.index.hour == k),'grid_power_import_fee'] = 7.5




# Example 3 - Time based tariff
# Prices for Ellevio AB Dalarna Södra (Fuse 16A, house)
fixed_charge = 25.5 # €/Month

df['grid_energy_import_fee'] = 9*0.001 # €/kWh
df['grid_energy_export_fee'] = 0 # €/kWh
df['grid_power_import_fee'] = 0 # €/kW-month
df['grid_power_export_fee'] = 0 # €/kW-month

# Set grid energy fee for different months, days and hours
for i in [1,2,3,11,12]:
    for j in list(range(0,5)):
        for k in list(range(8,22)):
            df.loc[(df.index.month == i) & (df.index.weekday == j) & (df.index.hour == k),'grid_energy_import_fee'] = 58*0.001


for i in [4,5,6,7,8,9,10]:
    df.loc[(df.index.month == i),'grid_energy_import_fee'] = 9*0.001




#%% Scenario 1: No PV/Battery/Heating
data_dict = {
        'demand': df['Electricity demand [kWh]'].to_list(),

        'energy_price_buy': (df['DAM Price [EUR/MWh]']/1000).to_list(),
        'energy_price_sell': (0.99*df['DAM Price [EUR/MWh]']/1000).to_list(),

        'grid_fixed_fee': fixed_charge,
        'grid_energy_import_fee': df['grid_energy_import_fee'].to_list(),
        'grid_energy_export_fee': df['grid_energy_export_fee'].to_list(),
        'grid_power_import_fee': df['grid_power_import_fee'].to_list(),
        'grid_power_export_fee': df['grid_power_export_fee'].to_list(),


        'emission_factor_grid': list(5*np.ones(len(df))),

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


#%% Scenario 2: PV/Battery sizing - No Heating
data_dict = {
        'invest_cost_battery': 20,
        'invest_cost_pv': 30,

        'battery_capacity_max': 5,
        'pv_capacity_max': 20,

        'pv_lifespan': 25,
        'battery_lifespan': 10,

        'discount_rate': 0.03,

        'demand': df['Electricity demand [kWh]'].to_list(),
        'generation_pu': df['PV 1kWp 180Azim 40Tilt Physical [kW]'].to_list(),

        'battery_min_level': 0.0,
        'battery_charge_max': 0.5,
        'battery_discharge_max': 0.5,
        'battery_efficiency_charge': 0.9,
        'battery_efficiency_discharge': 0.9,
        'battery_grid_charging': True,
        'bel_ini_level': 0.0,
        'bel_fin_level': 0.0,

        'energy_price_buy': (df['DAM Price [EUR/MWh]']/1000).to_list(),
        'energy_price_sell': (0.99*df['DAM Price [EUR/MWh]']/1000).to_list(),

        'grid_fixed_fee': fixed_charge,
        'grid_energy_import_fee': df['grid_energy_import_fee'].to_list(),
        'grid_energy_export_fee': df['grid_energy_export_fee'].to_list(),
        'grid_power_import_fee': df['grid_power_import_fee'].to_list(),
        'grid_power_export_fee': df['grid_power_export_fee'].to_list(),

        'emission_factor_grid': list(5*np.ones(len(df))),

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


#%% Scenario 3: PV/Battery/Heat pump sizing
data_dict = {
        'invest_cost_battery': 20,
        'invest_cost_pv': 30,
        'invest_cost_heat_pump': 10,

        'battery_capacity_max': 5,
        'pv_capacity_max': 20,
        'heat_pump_capacity_max': 2000,

        'pv_lifespan': 25,
        'battery_lifespan': 10,
        'heat_pump_lifespan': 25,

        'discount_rate': 0.03,

        'demand': df['Electricity demand [kWh]'].to_list(),
        'generation_pu': df['PV 1kWp 180Azim 40Tilt Physical [kW]'].to_list(),
        'heat_demand': df['Heat demand [kWh]'].to_list(),

        'battery_min_level': 0.0,
        'battery_charge_max': 0.5,
        'battery_discharge_max': 0.5,
        'battery_efficiency_charge': 0.9,
        'battery_efficiency_discharge': 0.9,
        'battery_grid_charging': True,
        'bel_ini_level': 0.0,
        'bel_fin_level': 0.0,

        'energy_price_buy': (df['DAM Price [EUR/MWh]']/1000).to_list(),
        'energy_price_sell': (0.99*df['DAM Price [EUR/MWh]']/1000).to_list(),

        'grid_fixed_fee': fixed_charge,
        'grid_energy_import_fee': df['grid_energy_import_fee'].to_list(),
        'grid_energy_export_fee': df['grid_energy_export_fee'].to_list(),
        'grid_power_import_fee': df['grid_power_import_fee'].to_list(),
        'grid_power_export_fee': df['grid_power_export_fee'].to_list(),

        'heat_capacity_factor': 0.1,

        'emission_factor_grid': list(5*np.ones(len(df))),

        'heat_pump_cop': 2.5,

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


#%% Scenario 4: PV/Battery/Fuel boiler sizing
data_dict = {
        'invest_cost_battery': 20,
        'invest_cost_pv': 30,
        'invest_cost_fuel_boiler': 10,

        'battery_capacity_max': 5,
        'pv_capacity_max': 20,
        'fuel_boiler_capacity_max': 500,

        'pv_lifespan': 25,
        'battery_lifespan': 10,
        'fuel_boiler_lifespan': 25,

        'discount_rate': 0.03,

        'demand': df['Electricity demand [kWh]'].to_list(),
        'generation_pu': df['PV 1kWp 180Azim 40Tilt Physical [kW]'].to_list(),
        'heat_demand': df['Heat demand [kWh]'].to_list(),

        'battery_min_level': 0.0,
        'battery_charge_max': 0.5,
        'battery_discharge_max': 0.5,
        'battery_efficiency_charge': 0.9,
        'battery_efficiency_discharge': 0.9,
        'battery_grid_charging': True,
        'bel_ini_level': 0.0,
        'bel_fin_level': 0.0,

        'energy_price_buy': (df['DAM Price [EUR/MWh]']/1000).to_list(),
        'energy_price_sell': (0.99*df['DAM Price [EUR/MWh]']/1000).to_list(),

        'grid_fixed_fee': fixed_charge,
        'grid_energy_import_fee': df['grid_energy_import_fee'].to_list(),
        'grid_energy_export_fee': df['grid_energy_export_fee'].to_list(),
        'grid_power_import_fee': df['grid_power_import_fee'].to_list(),
        'grid_power_export_fee': df['grid_power_export_fee'].to_list(),

        'fuel_price': list(20*np.ones(len(df))),

        'heat_capacity_factor': 0.1,

        'emission_factor_grid': list(5*np.ones(len(df))),
        'emission_factor_fuel': list(5*np.ones(len(df))),

        'fuel_boiler_efficiency': 0.9,

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
