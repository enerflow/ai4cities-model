from pyomo.environ import ConcreteModel
from pyomo.environ import Set,Param,Var,Objective,Constraint
from pyomo.environ import NonNegativeReals, Reals
from pyomo.environ import SolverFactory, minimize
from pyomo.environ import value
from pyomo.core.base.param import SimpleParam
import numpy as np
from sys import exit



def solve_model(model_instance, solver):
    if 'path' in solver:
        optimizer = SolverFactory(solver['name'], executable=solver['path'])
    else:
        optimizer = SolverFactory(solver['name'])

    optimizer.solve(model_instance, tee=True, keepfiles=False)

    return model_instance


def model_invest(model_data):


    model = ConcreteModel()

    ## SETS
    model.T = Set(dimen=1, ordered=True, initialize=model_data[None]['T']) # Periods
    model.M = Set(dimen=1, ordered=True, initialize=np.array(list(set(model_data[None]['month_order'].values())))) # Months


    ## PARAMETERS
    model.pwr_demand                    = Param(model.T, initialize=model_data[None]['power_demand'])
    model.pwr_gen                       = Param(model.T, initialize=model_data[None]['power_generation_pu'])
    model.ht_demand                     = Param(model.T, initialize=model_data[None]['heat_demand'])

    model.el_price_buy                  = Param(model.T, initialize=model_data[None]['electricity_price_buy'])
    model.el_price_sell                 = Param(model.T, initialize=model_data[None]['electricity_price_sell'])
    model.fl_price                      = Param(model.T, initialize=model_data[None]['fuel_price'])
    model.dh_price                      = Param(model.T, initialize=model_data[None]['district_heat_price'])
    
    model.grd_fee_fxd                   = Param(initialize=model_data[None]['grid_fee_fixed'])
    model.grd_fee_nrg_in                = Param(model.T, initialize=model_data[None]['grid_fee_energy_import'])
    model.grd_fee_nrg_out               = Param(model.T, initialize=model_data[None]['grid_fee_energy_export'])
    model.grd_fee_pwr_in                = Param(model.T, initialize=model_data[None]['grid_fee_power_import'])
    model.grd_fee_pwr_out               = Param(model.T, initialize=model_data[None]['grid_fee_power_export'])    

    model.invest_cost_pv                = Param(initialize=model_data[None]['invest_annual_unit_cost_pv'])
    model.invest_cost_bttr              = Param(initialize=model_data[None]['invest_annual_unit_cost_battery'])
    model.invest_cost_hp                = Param(initialize=model_data[None]['invest_annual_unit_cost_heat_pump'])    
    model.invest_cost_fb                = Param(initialize=model_data[None]['invest_annual_unit_cost_fuel_boiler'])
    model.invest_cost_eb                = Param(initialize=model_data[None]['invest_annual_unit_cost_electric_boiler'])
    model.invest_cost_tes               = Param(initialize=model_data[None]['invest_annual_unit_cost_thermal_storage'])
    model.invest_cost_dh                = Param(initialize=model_data[None]['invest_annual_unit_cost_district_heat'])
    
    model.heat_cap_over                 = Param(initialize=model_data[None]['heat_capacity_oversize'])
     
    model.pv_cap_max                    = Param(initialize=model_data[None]['pv_capacity_max'])
    model.pv_cap_min                    = Param(initialize=model_data[None]['pv_capacity_min'])
   
    model.dh_cap_max                    = Param(initialize=model_data[None]['district_heat_capacity_max'])
    model.dh_cap_min                    = Param(initialize=model_data[None]['district_heat_capacity_min'])
    
    model.hp_cap_max                    = Param(initialize=model_data[None]['heat_pump_capacity_max'])
    model.hp_cap_min                    = Param(initialize=model_data[None]['heat_pump_capacity_min'])
    model.hp_cop                        = Param(initialize=model_data[None]['heat_pump_cop'])
    
    model.eb_cap_max                    = Param(initialize=model_data[None]['electric_boiler_capacity_max'])
    model.eb_cap_min                    = Param(initialize=model_data[None]['electric_boiler_capacity_min'])
    model.eb_eff                        = Param(initialize=model_data[None]['electric_boiler_efficiency'])
    
    model.fb_cap_max                    = Param(initialize=model_data[None]['fuel_boiler_capacity_max'])
    model.fb_cap_min                    = Param(initialize=model_data[None]['fuel_boiler_capacity_min'])
    model.fb_eff                        = Param(initialize=model_data[None]['fuel_boiler_efficiency'])

    model.bttr_cap_max                  = Param(initialize=model_data[None]['battery_capacity_max'])
    model.bttr_cap_min                  = Param(initialize=model_data[None]['battery_capacity_min'])
    model.bttr_soc_min                  = Param(initialize=model_data[None]['battery_soc_min'])
    model.bttr_in_max                   = Param(initialize=model_data[None]['battery_charge_max'])
    model.bttr_out_max                  = Param(initialize=model_data[None]['battery_discharge_max'])
    model.bttr_eff_in                   = Param(initialize=model_data[None]['battery_efficiency_charge'])
    model.bttr_eff_out                  = Param(initialize=model_data[None]['battery_efficiency_discharge'])
    model.bttr_soc_ini                  = Param(initialize=model_data[None]['battery_soc_ini'])
    model.bttr_soc_fin                  = Param(initialize=model_data[None]['battery_soc_fin'])
    model.bttr_grid_in                  = Param(initialize=model_data[None]['battery_grid_charging'])
    
    model.tes_cap_max                   = Param(initialize=model_data[None]['thermal_storage_capacity_max'])
    model.tes_cap_min                   = Param(initialize=model_data[None]['thermal_storage_capacity_min'])
    model.tes_in_max                    = Param(initialize=model_data[None]['thermal_storage_charge_max'])
    model.tes_out_max                   = Param(initialize=model_data[None]['thermal_storage_discharge_max'])
    model.tes_loss                      = Param(initialize=model_data[None]['thermal_storage_losses'])
    model.tes_soc_ini                   = Param(initialize=model_data[None]['thermal_storage_soc_ini'])
    model.tes_soc_fin                   = Param(initialize=model_data[None]['thermal_storage_soc_fin'])
    
    model.emission_grid                 = Param(model.T, initialize=model_data[None]['specific_emission_grid'])
    model.emission_fuel                 = Param(model.T, initialize=model_data[None]['specific_emission_fuel'])
    model.emission_dh                   = Param(model.T, initialize=model_data[None]['specific_emission_district_heat'])
  
    model.dt                            = Param(initialize=model_data[None]['dt'])


    ## VARIABLES
    model.COST_INV                      = Var(within=Reals)
    model.COST_ENERGY                   = Var(model.T, within=Reals)
    model.COST_GRID_ENERGY_IMPORT       = Var(model.T, within=Reals)
    model.COST_GRID_ENERGY_EXPORT       = Var(model.T, within=Reals)
    model.COST_GRID_POWER_IMPORT        = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_POWER_EXPORT        = Var(model.T, within=NonNegativeReals)
    model.COST_GRID_POWER_IMPORT_MAX    = Var(model.M, within=Reals)
    model.COST_GRID_POWER_EXPORT_MAX    = Var(model.M, within=Reals)
    model.COST_GRID_FIXED               = Var(within=Reals)
    model.COST_FUEL                     = Var(model.T, within=Reals)
    model.COST_DH                       = Var(model.T, within=Reals)
    
    model.P_BUY                         = Var(model.T, within=NonNegativeReals)
    model.P_SELL                        = Var(model.T, within=NonNegativeReals) 
    model.P_GEN                         = Var(model.T, within=Reals)
    
    model.P_HP                          = Var(model.T, within=NonNegativeReals)
    model.P_EB                          = Var(model.T, within=NonNegativeReals)
    model.F_FB                          = Var(model.T, within=NonNegativeReals)
    
    model.Q_DH                          = Var(model.T, within=NonNegativeReals)
    model.Q_HP                          = Var(model.T, within=NonNegativeReals)
    model.Q_EB                          = Var(model.T, within=NonNegativeReals)
    model.Q_FB                          = Var(model.T, within=NonNegativeReals)

    model.BTTR_SOC                      = Var(model.T, within=NonNegativeReals)
    model.BTTR_IN                       = Var(model.T, within=NonNegativeReals)
    model.BTTR_OUT                      = Var(model.T, within=NonNegativeReals)
    
    model.TES_SOC                       = Var(model.T, within=NonNegativeReals)
    model.TES_IN                        = Var(model.T, within=NonNegativeReals)
    model.TES_OUT                       = Var(model.T, within=NonNegativeReals)

    model.PV_CAP                        = Var(within=NonNegativeReals, bounds=(model.pv_cap_min, model.pv_cap_max))
    model.BTTR_CAP                      = Var(within=NonNegativeReals, bounds=(model.bttr_cap_min, model.bttr_cap_max))
    model.HP_CAP                        = Var(within=NonNegativeReals, bounds=(model.hp_cap_min, model.hp_cap_max))
    model.FB_CAP                        = Var(within=NonNegativeReals, bounds=(model.fb_cap_min, model.fb_cap_max))
    model.EB_CAP                        = Var(within=NonNegativeReals, bounds=(model.eb_cap_min, model.eb_cap_max))
    model.DH_CAP                        = Var(within=NonNegativeReals, bounds=(model.dh_cap_min, model.dh_cap_max))
    model.TES_CAP                       = Var(within=NonNegativeReals, bounds=(model.tes_cap_min, model.tes_cap_max))

    model.CO2_GRID                      = Var(model.T, within=Reals)
    model.CO2_FUEL                      = Var(model.T, within=Reals)
    model.CO2_DH                        = Var(model.T, within=Reals)


    ## OBJECTIVE
    # Minimize cost
    def total_cost(model):
        return  model.COST_INV \
        + sum(model.COST_ENERGY[t] for t in model.T) \
        + sum(model.COST_GRID_ENERGY_IMPORT[t] + model.COST_GRID_ENERGY_EXPORT[t] for t in model.T) \
        + sum(model.COST_GRID_POWER_IMPORT_MAX[m] + model.COST_GRID_POWER_EXPORT_MAX[m] for m in model.M) \
        + model.COST_GRID_FIXED \
        + sum(model.COST_FUEL[t] for t in model.T) \
        + sum(model.COST_DH[t] for t in model.T)
    model.total_cost = Objective(rule=total_cost, sense=minimize)



    ## CONSTRAINTS
    # Investment cost
    def investment_cost(model):
        return model.COST_INV == model.invest_cost_pv*model.PV_CAP \
                                + model.invest_cost_bttr*model.BTTR_CAP \
                                + model.invest_cost_hp*model.HP_CAP \
                                + model.invest_cost_eb*model.EB_CAP \
                                + model.invest_cost_fb*model.FB_CAP \
                                + model.invest_cost_dh*model.DH_CAP \
                                + model.invest_cost_tes*model.TES_CAP                            
    model.investment_cost = Constraint(rule=investment_cost)
    
    
    # Energy cost
    def energy_cost(model, t):
        return model.COST_ENERGY[t] == model.el_price_buy[t]*model.P_BUY[t]*model.dt - model.el_price_sell[t]*model.P_SELL[t]*model.dt
    model.energy_cost = Constraint(model.T, rule=energy_cost)


    # Grid fixed cost
    def grid_fixed_cost(model):
        return model.COST_GRID_FIXED == model.grd_fee_fxd*len(model.M)
    model.grid_fixed_cost = Constraint(rule=grid_fixed_cost)
    
    # Grid energy import cost
    def grid_energy_import_cost(model, t):
        return model.COST_GRID_ENERGY_IMPORT[t] == model.grd_fee_nrg_in[t]*model.P_BUY[t]*model.dt
    model.grid_energy_import_cost = Constraint(model.T, rule=grid_energy_import_cost)
    
    # Grid energy export cost
    def grid_energy_export_cost(model, t):
        return model.COST_GRID_ENERGY_EXPORT[t] == model.grd_fee_nrg_out[t]*model.P_SELL[t]*model.dt
    model.grid_energy_export_cost = Constraint(model.T, rule=grid_energy_export_cost)

    # Grid power import cost
    def grid_power_import_cost(model, t):
        return model.COST_GRID_POWER_IMPORT[t] >= model.grd_fee_pwr_in[t]*(model.P_BUY[t]-model.P_SELL[t])
    model.grid_power_import_cost = Constraint(model.T, rule=grid_power_import_cost)
    
    # Grid power export cost
    def grid_power_export_cost(model, t):
        return model.COST_GRID_POWER_EXPORT[t] >= model.grd_fee_pwr_out[t]*(model.P_SELL[t]-model.P_BUY[t])
    model.grid_power_export_cost = Constraint(model.T, rule=grid_power_export_cost)

    # Max grid import cost
    def max_grid_power_import_cost(model, t):
        return model.COST_GRID_POWER_IMPORT_MAX[model_data[None]['month_order'][t]] >= model.COST_GRID_POWER_IMPORT[t]
    model.max_grid_power_import_cost = Constraint(model.T, rule=max_grid_power_import_cost)

    # Max grid export cost
    def max_grid_power_export_cost(model, t):
        return model.COST_GRID_POWER_EXPORT_MAX[model_data[None]['month_order'][t]] >= model.COST_GRID_POWER_EXPORT[t]
    model.max_grid_power_export_cost = Constraint(model.T, rule=max_grid_power_export_cost)


    # Fuel boiler cost
    def fuel_cost(model, t):
        return model.COST_FUEL[t] == model.fl_price[t]*model.F_FB[t]*model.dt
    model.fuel_cost = Constraint(model.T, rule=fuel_cost)
    
    # District heating cost
    def dh_cost(model, t):
        return model.COST_DH[t] == model.dh_price[t]*model.Q_DH[t]*model.dt
    model.dh_cost = Constraint(model.T, rule=dh_cost)
    

    # Power generation
    def power_generation(model, t):
        return model.P_GEN[t] == model.pwr_gen[t]*model.PV_CAP
    model.power_generation = Constraint(model.T, rule=power_generation)
    

    # Power balance
    def energy_balance(model, t):
        return model.P_SELL[t] - model.P_BUY[t] == model.P_GEN[t] + model.BTTR_OUT[t] - model.BTTR_IN[t] - model.pwr_demand[t] - model.P_HP[t] - model.P_EB[t]
    model.energy_balance = Constraint(model.T, rule=energy_balance)


    # Battery energy balance
    def battery_soc(model, t):
        if t==model.T.first():
            return model.BTTR_SOC[t] - model.bttr_soc_ini*model.BTTR_CAP == model.bttr_eff_in*model.BTTR_IN[t]*model.dt  - (1/model.bttr_eff_out)*model.BTTR_OUT[t]*model.dt
        else:
            return model.BTTR_SOC[t] - model.BTTR_SOC[model.T.prev(t)] == model.bttr_eff_in*model.BTTR_IN[t]*model.dt  - (1/model.bttr_eff_out)*model.BTTR_OUT[t]*model.dt
    model.battery_soc = Constraint(model.T, rule=battery_soc)

    # Battery SOC upper limit
    def battery_soc_upper_limit(model, t):
        return model.BTTR_SOC[t] <= model.BTTR_CAP
    model.battery_soc_upper_limit = Constraint(model.T, rule=battery_soc_upper_limit)

    # Battery SOC lower limit
    def battery_soc_lower_limit(model, t):
        return model.BTTR_SOC[t] >= model.bttr_soc_min*model.BTTR_CAP
    model.battery_soc_lower_limit = Constraint(model.T, rule=battery_soc_lower_limit)

    # Battery charge upper limit
    def battery_charge_upper_limit(model, t):
        return model.BTTR_IN[t] <= model.bttr_in_max*model.BTTR_CAP
    model.battery_charge_upper_limit = Constraint(model.T, rule=battery_charge_upper_limit)

    def battery_discharge_upper_limit(model, t):
        return model.BTTR_OUT[t] <= model.bttr_out_max*model.BTTR_CAP
    model.battery_discharge_upper_limit = Constraint(model.T, rule=battery_discharge_upper_limit)

    # Battery charging from grid
    def no_grid_charging(model, t):
        if value(model.bttr_grid_in) == False:
            return model.P_BUY[t] <= model.pwr_demand[t] + model.P_HP[t] + model.P_EB[t]   
        else:
            return model.P_BUY[t] <= model.pwr_demand[t] + model.P_HP[t] + model.P_EB[t] + model.BTTR_IN[t]
    model.no_grid_charging = Constraint(model.T, rule=no_grid_charging)


    # Fuel boiler heat generation - fuel consumption
    def fuel_boiler_gen(model, t):
        return model.Q_FB[t] == model.fb_eff*model.F_FB[t]
    model.fuel_boiler_gen = Constraint(model.T, rule=fuel_boiler_gen)

    # Heat pump heat generation - power consumption
    def heat_pump_gen(model, t):
        return model.Q_HP[t] == model.hp_cop*model.P_HP[t]
    model.heat_pump_gen = Constraint(model.T, rule=heat_pump_gen)
    
    # Electric boiler heat generation - power consumption
    def electric_boiler_gen(model, t):
        return model.Q_EB[t] == model.eb_eff*model.P_EB[t]
    model.electric_boiler_gen = Constraint(model.T, rule=electric_boiler_gen)
    
    
    # Fuel boiler capacity
    def fuel_boiler_cap(model, t):
        return (1 + model.heat_cap_over)*model.Q_FB[t] <= model.FB_CAP
    model.fuel_boiler_cap = Constraint(model.T, rule=fuel_boiler_cap)
    
    # Heat pump capacity
    def heat_pump_cap(model, t):
        return (1 + model.heat_cap_over)*model.Q_HP[t] <= model.HP_CAP
    model.heat_pump_cap = Constraint(model.T, rule=heat_pump_cap)
    
    # Electric boiler capacity
    def electric_boiler_cap(model, t):
        return (1 + model.heat_cap_over)*model.Q_EB[t] <= model.EB_CAP
    model.electric_boiler_cap = Constraint(model.T, rule=electric_boiler_cap)
    
    # District heating capacity
    def district_heating_cap(model, t):
        return (1 + model.heat_cap_over)*model.Q_DH[t] <= model.DH_CAP
    model.district_heating_cap = Constraint(model.T, rule=district_heating_cap)
    
    
    # Heat balance
    def heat_balance(model, t):
        return model.Q_FB[t] + model.Q_HP[t] + model.Q_EB[t] + model.Q_DH[t] == model.ht_demand[t] - model.TES_OUT[t] + model.TES_IN[t]
    model.heat_balance = Constraint(model.T, rule=heat_balance)


    # Heat storage energy balance
    def tes_soc(model, t):
        if t==model.T.first():
            return model.TES_SOC[t] - (1-model.tes_loss)*model.tes_soc_ini*model.TES_CAP == model.TES_IN[t]*model.dt - model.TES_OUT[t]*model.dt
        else:
            return model.TES_SOC[t] - (1-model.tes_loss)*model.TES_SOC[model.T.prev(t)] == model.TES_IN[t]*model.dt - model.TES_OUT[t]*model.dt
    model.tes_soc = Constraint(model.T, rule=tes_soc)
	
	# Heat storage upper limit
    def tes_upper_bound(model, t):
        return model.TES_SOC[t] <= model.TES_CAP
    model.tes_upper_bound = Constraint(model.T, rule=tes_upper_bound)
	
	# Heat storage charge upper limit
    def tes_charge_limit(model, t):
        return model.TES_IN[t] <= model.tes_in_max*model.TES_CAP
    model.tes_charge_limit = Constraint(model.T, rule=tes_charge_limit)

	# Heat storage discharge upper limit
    def tes_discharge_limit(model, t):
        return model.TES_OUT[t] <= model.tes_out_max*model.TES_CAP
    model.tes_discharge_limit = Constraint(model.T, rule=tes_discharge_limit)


    # Grid CO2 emissions
    def carbon_emissions_grid(model, t):
        return model.CO2_GRID[t] == (model.P_BUY[t] - model.P_SELL[t])*model.emission_grid[t]*model.dt
    model.carbon_emissions_grid = Constraint(model.T, rule=carbon_emissions_grid)
    
    # Fuel CO2 emissions
    def carbon_emissions_fuel(model, t):
        return model.CO2_FUEL[t] == model.F_FB[t]*model.emission_fuel[t]*model.dt
    model.carbon_emissions_fuel = Constraint(model.T, rule=carbon_emissions_fuel) 
    
    # District heating CO2 emissions
    def carbon_emissions_dh(model, t):
        return model.CO2_DH[t] == model.Q_DH[t]*model.emission_dh[t]*model.dt
    model.carbon_emissions_dh = Constraint(model.T, rule=carbon_emissions_dh) 



    # Fix battery soc in the last period
    if value(model.bttr_soc_fin) > 0:
        model.BTTR_SOC[model.T.last()].fix(model.bttr_soc_fin*model.BTTR_CAP)
        
        
    # Fix tes soc in the last period
    if value(model.tes_soc_fin) > 0:
        model.TES_SOC[model.T.last()].fix(model.tes_soc_fin*model.TES_CAP)
        
     
    return model


def model_invest_input(data):
    
    
    def annuity_rate(discount_rate, salvage_rate, life_span):
        
        anrate = (1-salvage_rate)*discount_rate*(1+discount_rate)**life_span/((1+discount_rate)**life_span-1)
        return anrate

    if 'power_generation_pu' in data:
        periods = np.arange(1, len(data['power_generation_pu'])+1)
    elif 'power_demand' in data:
        periods = np.arange(1, len(data['power_demand'])+1)
    elif 'heat_demand' in data:
        periods = np.arange(1, len(data['heat_demand'])+1)

    power_generation_pu = dict(zip(periods,  data['power_generation_pu'])) if 'power_generation_pu' in data else dict(zip(periods,  [0] * len(periods)))
    power_demand = dict(zip(periods,  data['power_demand'])) if 'power_demand' in data else dict(zip(periods,  [0] * len(periods)))
    heat_demand = dict(zip(periods,  data['heat_demand'])) if 'heat_demand' in data else dict(zip(periods,  [0] * len(periods)))

    electricity_price_buy = dict(zip(periods,  data['electricity_price_buy'])) if 'electricity_price_buy' in data else dict(zip(periods,  [0] * len(periods)))
    electricity_price_sell = dict(zip(periods,  data['electricity_price_sell'])) if 'electricity_price_sell' in data else dict(zip(periods,  [0] * len(periods)))
    
    grid_fee_fixed = data['grid_fee_fixed'] if 'grid_fee_fixed' in data else 0.0
    grid_fee_energy_import = dict(zip(periods,  data['grid_fee_energy_import'])) if 'grid_fee_energy_import' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_energy_export = dict(zip(periods,  data['grid_fee_energy_export'])) if 'grid_fee_energy_export' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_power_import = dict(zip(periods,  data['grid_fee_power_import'])) if 'grid_fee_power_import' in data else dict(zip(periods,  [0] * len(periods)))
    grid_fee_power_export = dict(zip(periods,  data['grid_fee_power_export'])) if 'grid_fee_power_export' in data else dict(zip(periods,  [0] * len(periods)))

    fuel_price = dict(zip(periods,  data['fuel_price'])) if 'fuel_price' in data else dict(zip(periods,  [0] * len(periods)))
    district_heat_price = dict(zip(periods,  data['district_heat_price'])) if 'district_heat_price' in data else dict(zip(periods,  [0] * len(periods)))

    discount_rate = data['discount_rate'] if 'discount_rate' in data else 0.0
    salvage_rate = data['salvage_rate'] if 'salvage_rate' in data else 0.0

    pv_lifespan = data['pv_lifespan'] if 'pv_lifespan' in data else 25
    battery_lifespan = data['battery_lifespan'] if 'battery_lifespan' in data else 10
    heat_pump_lifespan = data['heat_pump_lifespan'] if 'heat_pump_lifespan' in data else 25
    electric_boiler_lifespan = data['electric_boiler_lifespan'] if 'electric_boiler_lifespan' in data else 25
    fuel_boiler_lifespan = data['fuel_boiler_lifespan'] if 'fuel_boiler_lifespan' in data else 25
    district_heat_lifespan = data['district_heat_lifespan'] if 'district_heat_lifespan' in data else 50
    thermal_storage_lifespan = data['thermal_storage_lifespan'] if 'thermal_storage_lifespan' in data else 30

    annuity_rate_pv = annuity_rate(discount_rate, salvage_rate, pv_lifespan)
    annuity_rate_battery = annuity_rate(discount_rate, salvage_rate, battery_lifespan)
    annuity_rate_heat_pump = annuity_rate(discount_rate, salvage_rate, heat_pump_lifespan)
    annuity_rate_electric_boiler = annuity_rate(discount_rate, salvage_rate, electric_boiler_lifespan)
    annuity_rate_fuel_boiler = annuity_rate(discount_rate, salvage_rate, fuel_boiler_lifespan)
    annuity_rate_district_heat = annuity_rate(discount_rate, salvage_rate, district_heat_lifespan)
    annuity_rate_thermal_storage = annuity_rate(discount_rate, salvage_rate, thermal_storage_lifespan)

    invest_annual_unit_cost_pv = data['invest_unit_cost_pv']*annuity_rate_pv if 'invest_unit_cost_pv' in data else 0.0
    invest_annual_unit_cost_battery = data['invest_unit_cost_battery']*annuity_rate_battery if 'invest_unit_cost_battery' in data else 0.0
    invest_annual_unit_cost_heat_pump = data['invest_unit_cost_heat_pump']*annuity_rate_heat_pump if 'invest_unit_cost_heat_pump' in data else 0.0
    invest_annual_unit_cost_electric_boiler = data['invest_unit_cost_electric_boiler']*annuity_rate_electric_boiler if 'invest_unit_cost_electric_boiler' in data else 0.0
    invest_annual_unit_cost_fuel_boiler = data['invest_unit_cost_fuel_boiler']*annuity_rate_fuel_boiler if 'invest_unit_cost_fuel_boiler' in data else 0.0
    invest_annual_unit_cost_district_heat = data['invest_unit_cost_district_heat']*annuity_rate_district_heat if 'invest_unit_cost_district_heat' in data else 0.0
    invest_annual_unit_cost_thermal_storage = data['invest_unit_cost_thermal_storage']*annuity_rate_thermal_storage if 'invest_unit_cost_thermal_storage' in data else 0.0
    
    heat_capacity_oversize = data['heat_capacity_oversize'] if 'heat_capacity_oversize' in data else 0.0
    
    pv_capacity_max = data['pv_capacity_max'] if 'pv_capacity_max' in data else 0.0
    pv_capacity_min = data['pv_capacity_min'] if 'pv_capacity_min' in data else 0.0
    
    district_heat_capacity_max = data['district_heat_capacity_max'] if 'district_heat_capacity_max' in data else 0.0
    district_heat_capacity_min = data['district_heat_capacity_min'] if 'district_heat_capacity_min' in data else 0.0

    heat_pump_capacity_max = data['heat_pump_capacity_max'] if 'heat_pump_capacity_max' in data else 0.0
    heat_pump_capacity_min = data['heat_pump_capacity_min'] if 'heat_pump_capacity_min' in data else 0.0
    heat_pump_cop = data['heat_pump_cop'] if 'heat_pump_cop' in data else 3.0
    
    electric_boiler_capacity_max = data['electric_boiler_capacity_max'] if 'electric_boiler_capacity_max' in data else 0.0
    electric_boiler_capacity_min = data['electric_boiler_capacity_min'] if 'electric_boiler_capacity_min' in data else 0.0
    electric_boiler_efficiency = data['electric_boiler_efficiency'] if 'electric_boiler_efficiency' in data else 0.0
    
    fuel_boiler_capacity_max = data['fuel_boiler_capacity_max'] if 'fuel_boiler_capacity_max' in data else 0.0
    fuel_boiler_capacity_min = data['fuel_boiler_capacity_min'] if 'fuel_boiler_capacity_min' in data else 0.0
    fuel_boiler_efficiency = data['fuel_boiler_efficiency'] if 'fuel_boiler_efficiency' in data else 0.0
    
    battery_capacity_max = data['battery_capacity_max'] if 'battery_capacity_max' in data else 0.0
    battery_capacity_min = data['battery_capacity_min'] if 'battery_capacity_min' in data else 0.0
    battery_soc_min = data['battery_soc_min'] if 'battery_soc_min' in data else 0.0
    battery_charge_max = data['battery_charge_max'] if 'battery_charge_max' in data else 1.0
    battery_discharge_max = data['battery_discharge_max'] if 'battery_discharge_max' in data else 1.0
    battery_efficiency_charge = data['battery_efficiency_charge'] if 'battery_efficiency_charge' in data else 0.95
    battery_efficiency_discharge = data['battery_efficiency_discharge'] if 'battery_efficiency_discharge' in data else 0.95
    battery_soc_ini = data['battery_soc_ini'] if 'battery_soc_ini' in data else 0.0
    battery_soc_fin = data['battery_soc_fin'] if 'battery_soc_fin' in data else 0.0 
    battery_grid_charging = data['battery_grid_charging'] if 'battery_grid_charging' in data else True
      
    thermal_storage_capacity_max = data['thermal_storage_capacity_max'] if 'thermal_storage_capacity_max' in data else 0.0
    thermal_storage_capacity_min = data['thermal_storage_capacity_min'] if 'thermal_storage_capacity_min' in data else 0.0
    thermal_storage_charge_max = data['thermal_storage_charge_max'] if 'thermal_storage_charge_max' in data else 1.0
    thermal_storage_discharge_max = data['thermal_storage_discharge_max'] if 'thermal_storage_discharge_max' in data else 1.0
    thermal_storage_losses = data['thermal_storage_losses'] if 'thermal_storage_losses' in data else 0.05
    thermal_storage_soc_ini = data['thermal_storage_soc_ini'] if 'thermal_storage_soc_ini' in data else 0.0
    thermal_storage_soc_fin = data['thermal_storage_soc_fin'] if 'thermal_storage_soc_fin' in data else 0.0

    specific_emission_grid = dict(zip(periods,  data['specific_emission_grid'])) if 'specific_emission_grid' in data else dict(zip(periods,  [0] * len(periods)))
    specific_emission_fuel = dict(zip(periods,  data['specific_emission_fuel'])) if 'specific_emission_fuel' in data else dict(zip(periods,  [0] * len(periods)))
    specific_emission_district_heat = dict(zip(periods,  data['specific_emission_district_heat'])) if 'specific_emission_district_heat' in data else dict(zip(periods,  [0] * len(periods)))

    dt = data['dt'] if 'dt' in data else 1.0
    month_order = dict(zip(periods,  data['month_order'])) if 'month_order' in data else dict(zip(periods,  [1] * len(periods)))


    min_feasible_heat_capacity =  max(heat_demand.values())*(1+heat_capacity_oversize)    
   
    if heat_pump_capacity_max  + electric_boiler_capacity_max + fuel_boiler_capacity_max  + district_heat_capacity_max < min_feasible_heat_capacity:
        
        print('Total maximum heat capacity is too low. Increase at least to: ', min_feasible_heat_capacity)
        exit()
        

    # Create model data input dictionary
    model_data = {None: {
        'T': periods,
        
        'invest_annual_unit_cost_battery': invest_annual_unit_cost_battery,
        'invest_annual_unit_cost_pv': invest_annual_unit_cost_pv,
        'invest_annual_unit_cost_heat_pump': invest_annual_unit_cost_heat_pump,
        'invest_annual_unit_cost_electric_boiler': invest_annual_unit_cost_electric_boiler,
        'invest_annual_unit_cost_fuel_boiler': invest_annual_unit_cost_fuel_boiler,
        'invest_annual_unit_cost_district_heat': invest_annual_unit_cost_district_heat,
        'invest_annual_unit_cost_thermal_storage': invest_annual_unit_cost_thermal_storage,
        
        'heat_capacity_oversize': heat_capacity_oversize,
        
        'power_generation_pu': power_generation_pu,
        'power_demand': power_demand,
        'heat_demand': heat_demand,
        
        'electricity_price_buy': electricity_price_buy,
        'electricity_price_sell': electricity_price_sell,
        'fuel_price': fuel_price,
        'district_heat_price': district_heat_price,
        
        'grid_fee_fixed': grid_fee_fixed,
        'grid_fee_energy_import': grid_fee_energy_import,
        'grid_fee_energy_export': grid_fee_energy_export,
        'grid_fee_power_import': grid_fee_power_import,
        'grid_fee_power_export': grid_fee_power_export,
        
        'pv_capacity_max': pv_capacity_max,
        'pv_capacity_min': pv_capacity_min,

        'battery_capacity_max': battery_capacity_max,
        'battery_capacity_min': battery_capacity_min,
        'battery_soc_min': battery_soc_min,
        'battery_charge_max': battery_charge_max,
        'battery_discharge_max': battery_discharge_max,
        'battery_efficiency_charge': battery_efficiency_charge,
        'battery_efficiency_discharge': battery_efficiency_discharge,
        'battery_grid_charging': battery_grid_charging,
        'battery_soc_ini': battery_soc_ini,
        'battery_soc_fin': battery_soc_fin,
        
        'district_heat_capacity_max': district_heat_capacity_max,
        'district_heat_capacity_min': district_heat_capacity_min,

        'heat_pump_capacity_max': heat_pump_capacity_max,
        'heat_pump_capacity_min': heat_pump_capacity_min,
        'heat_pump_cop': heat_pump_cop,
        
        'electric_boiler_capacity_max': electric_boiler_capacity_max,
        'electric_boiler_capacity_min': electric_boiler_capacity_min,
        'electric_boiler_efficiency': electric_boiler_efficiency,
        
        'fuel_boiler_capacity_max': fuel_boiler_capacity_max,
        'fuel_boiler_capacity_min': fuel_boiler_capacity_min,
        'fuel_boiler_efficiency': fuel_boiler_efficiency,
        
        'thermal_storage_capacity_max': thermal_storage_capacity_max,
        'thermal_storage_capacity_min': thermal_storage_capacity_min,
        'thermal_storage_charge_max': thermal_storage_charge_max,
        'thermal_storage_discharge_max': thermal_storage_discharge_max,
        'thermal_storage_losses': thermal_storage_losses,
        'thermal_storage_soc_ini': thermal_storage_soc_ini,
        'thermal_storage_soc_fin': thermal_storage_soc_fin,

        'specific_emission_grid': specific_emission_grid,
        'specific_emission_fuel': specific_emission_fuel,
        'specific_emission_district_heat': specific_emission_district_heat,
        
        'month_order': month_order,
        'dt': dt,
    }}

    return model_data


def model_invest_results(solution):
    
    s = dict()
    
    s['cost_total'] = solution.total_cost()
    
    s['cost_investment_annual'] = value(solution.COST_INV)
    s['cost_energy'] = value(solution.COST_ENERGY[:])
    s['cost_grid_energy_import'] = value(solution.COST_GRID_ENERGY_IMPORT[:])
    s['cost_grid_energy_export'] = value(solution.COST_GRID_ENERGY_EXPORT[:])
    s['cost_grid_power_import'] = value(solution.COST_GRID_POWER_IMPORT_MAX[:])
    s['cost_grid_power_export'] = value(solution.COST_GRID_POWER_EXPORT_MAX[:])
    s['cost_grid_power_fixed'] = value(solution.COST_GRID_FIXED)
    
    s['cost_fuel_heating'] = value(solution.COST_FUEL[:])
    s['cost_district_heating'] = value(solution.COST_DH[:])
        
    s['power_buy'] = value(solution.P_BUY[:])
    s['power_sell'] = value(solution.P_SELL[:])
    s['power_generation'] = value(solution.P_GEN[:])
    s['power_consumption_heat_pump'] = value(solution.P_HP[:])
    s['power_consumption_electric_boiler'] = value(solution.P_EB[:])
    s['heat_generation_heat_pump'] = value(solution.Q_HP[:])
    s['heat_generation_electric_boiler'] = value(solution.Q_EB[:])
    s['heat_generation_fuel_boiler'] = value(solution.Q_FB[:])
    s['heat_consumption_district'] = value(solution.Q_DH[:])
    s['fuel_consumption_boiler'] = value(solution.F_FB[:])
    
    s['battery_soc'] = value(solution.BTTR_SOC[:])
    s['battery_charge'] = value(solution.BTTR_IN[:])
    s['battery_discharge'] = value(solution.BTTR_OUT[:])
    
    s['thermal_storage_soc'] = value(solution.TES_SOC[:])
    s['thermal_storage_charge'] = value(solution.TES_IN[:])
    s['thermal_storage_discharge'] = value(solution.TES_OUT[:])

    s['carbon_emissions_grid'] = value(solution.CO2_GRID[:])
    s['carbon_emissions_fuel'] = value(solution.CO2_FUEL[:])
    s['carbon_emissions_district'] = value(solution.CO2_DH[:])
    
    s['capacity_pv'] = value(solution.PV_CAP)
    s['capacity_battery'] = value(solution.BTTR_CAP)
    s['capacity_heat_pump'] = value(solution.HP_CAP)
    s['capacity_electric_boiler'] = value(solution.EB_CAP)
    s['capacity_fuel_boiler'] = value(solution.FB_CAP)
    s['capacity_district'] = value(solution.DH_CAP)
    s['capacity_thermal_storage'] = value(solution.TES_CAP)

    return s