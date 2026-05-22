# AoFCE Coursework
# 18/12/2025
# Husaini Zaidan [06020282], #Zaid Janjua [02086052]

#%% CELL 1 - Import libraries and data
from pyomo.environ import *

#%% CELL 2 - Setup the Pyomo Solver
# Simultaenous-Method Group Contributions (Hukkerikar et al., Rayer et al.)
TB, TM, HD, HP, HH, VM, CP, VAL, MW = range(9)

groups = [
    "CH3","CH2","CH","C",
    "CH2=CH","CH=CH","CH2=C","CH=C","C=C",
    "OH","COOH","CH3CO","CH2CO","CHO","CHCOO","HCOO",
    "CH3O","CH2O","CH-O",
    "CH2NH2","CHNH2","CH3NH","CH2NH","CHNH",
    "CH3N","CH2N",
    "CONH2","CONHCH3","CON(CH3)2",
    "OCH2CH2OH","OCH2CHOH",
]

# each row: [Tb1, Tm1, Hd1, Hp1, Hh1, Vm1, Cp1, Valency, mw]
data = [
    [0.9218, 0.7555,  7.5983,  2.3037,  2.2105, 0.0238,  50.82, 1, 15.03],  # CH3
    [0.578,  0.2966, -0.0023, -0.1664, -0.215, 0.0166,  33.5,  2, 14.03],  # CH2
    [-0.1189,-0.596, -7.539,  -3.3851, -2.6826,0.0084,  41.11, 3, 13.02],  # CH
    [-0.6495,-0.3679,-15.6455,-5.1979, -6.4821,-0.0015, 2.48,  4, 12.01],  # C
    [1.4953, 1.043,  7.7504,  3.6752,  2.7673, 0.0333,  57.97, 1, 27.04],  # CH2=CH
    [1.2001, 0.66,   0.4284,  3.0492,  0.8631, 0.0244,  48.94, 2, 26.04],  # CH=CH
    [1.0308, 0.3327, 0.1956,  2.3059,  1.0623, 0.023,   20.21, 2, 26.04],  # CH2=C
    [0.7646,-0.3944,-7.0086,  1.279,  -0.1204,0.0108,  11.18, 3, 25.03],  # CH=C
    [0.408, -0.9826,-14.616, -1.459, -2.9995,-0.0021,-26.58, 4, 24.02],  # C=C
    [2.2476, 3.2424, 8.0503,  5.2379, 11.8005,0.0042,  59.74, 1, 17.01],  # OH
    [3.9741, 6.6986, 8.4172,  3.14,   7.5917,0.0207,  60.49, 1, 45.02],  # COOH
    [2.6907, 3.2535, 8.1107,  6.3823, 3.4394,0.0347,  91.12, 1, 43.04],  # CH3CO
    [1.9665, 2.8589, 0.5371,  1.2706,-0.0788,0.0283,  73.8,  2, 42.04],  # CH2CO
    [2.1021, 2.9059, 7.8411,  7.8726, 5.3761,0.0167,  81.41, 1, 41.03],  # CHO
    [1.4466, 0.9772,-7.2639, -2.8697,-1.1517,0.0284, 101.6,  3, 57.03],  # CHCOO
    [2.2249, 2.2831, 7.923,   4.8158, 6.8448,0.0256,  60.49, 1, 45.02],  # HCOO
    [1.584,  1.5577, 7.6317,  3.2154, 3.3464,0.0281,  71.01, 1, 31.03],  # CH3O
    [0.975,  0.6741, 0.1706,  0.5137, 0.8246,0.0228,  53.69, 2, 30.03],  # CH2O
    [0.3272,-0.0101,-7.6174, -2.7093,-2.1543,0.0205,  61.3,  3, 29.02],  # CH-O
    [2.264,  3.349,  8.1995,  5.2101, 6.7984,0.0262,  92.8,  1, 30.05],  # CH2NH2
    [1.4372,36.2974,-0.3812,  0.5616, 2.8953,0.0214, 100.41, 2, 29.04],  # CHNH2
    [1.986,  2.7394, 7.7307,  2.5065, 7.2551,0.0279,  96.27, 1, 30.05],  # CH3NH
    [1.269,  2.0378, 0.0223, -0.7159, 1.4183,0.0246,  78.95, 2, 29.04],  # CH2NH
    [0.594,  1.3226,-7.5377, -4.6694,-2.2824,0.0182,  86.56, 3, 28.04],  # CHNH
    [0.999,  0.8482, 0.3494, -0.425,  2.4585,0.0265,  52.97, 2, 29.04],  # CH3N
    [0.3324,-0.4084,-6.7232, -0.7354,-7.3014,0.019,   35.65, 3, 28.04],  # CH2N
    [5.2882,11.8818, 9.0888,  9.0329,10.6398,0.0147,  99.6,  1, 44.04],  # CONH2
    [4.2196, 6.5932, 7.4017,  1.9963, 3.4895,0.038,  136.57, 1, 58.06],  # CONHCH3
    [4.7548, 3.7173, 9.004,   8.9127, 6.8046,0.0533, 144.09, 1, 72.09],  # CON(CH3)2
    [3.8181, 2.4297, 8.3027,  5.7113, 9.9562,0.0417, 146.93, 2, 61.06],  # OCH2CH2OH
    [2.9869, 1.7527, 0.0881,  0.7182, 4.8512,0.0392, 154.54, 3, 60.05],  # OCH2CHOH
]

# build dicts from matrix
C_Tb   = {g: r[TB]  for g, r in zip(groups, data)}
C_Tm   = {g: r[TM]  for g, r in zip(groups, data)}
C_D    = {g: r[HD]  for g, r in zip(groups, data)}
C_P    = {g: r[HP]  for g, r in zip(groups, data)}
C_H    = {g: r[HH]  for g, r in zip(groups, data)}
C_Vm   = {g: r[VM]  for g, r in zip(groups, data)}
C_Cp   = {g: r[CP]  for g, r in zip(groups, data)}
valency= {g: r[VAL] for g, r in zip(groups, data)}
mws    = {g: r[MW]  for g, r in zip(groups, data)}

# Model Definition
model = ConcreteModel(name="CAMD")

# Sets
model.Groups = Set(initialize=groups)

# Params
model.C_Tb = Param(model.Groups, initialize=C_Tb)
model.C_Tm = Param(model.Groups, initialize=C_Tm)
model.C_D  = Param(model.Groups, initialize=C_D)
model.C_P  = Param(model.Groups, initialize=C_P)
model.C_H  = Param(model.Groups, initialize=C_H)
model.C_Vm = Param(model.Groups, initialize=C_Vm)
model.C_Cp = Param(model.Groups, initialize=C_Cp)
model.valency = Param(model.Groups, initialize=valency)
model.mws     = Param(model.Groups, initialize=mws)

# Scalar Parameters
model.Tb0 = Param(initialize=244.5165)
model.Tm0 = Param(initialize=143.5706)
model.Vm0 = Param(initialize=0.0160)
model.R0  = Param(initialize=4.0)

# Target/Reference Parameters (CO2)
model.delta_d_CO2 = Param(initialize=15.6)
model.delta_p_CO2 = Param(initialize=5.2)
model.delta_h_CO2 = Param(initialize=5.8)

# Objective Scaling
RED_min, RED_max   = 0, 20
rho_min, rho_max   = 900, 1300      # kg/m3
Cp_min,  Cp_max    = 83, 405        # J/mol·K

# Weight Vectors (set to mutable for Q3)
model.w1 = Param(initialize=1,mutable=True)
model.w2 = Param(initialize=1,mutable=True)
model.w3 = Param(initialize=1,mutable=True)

# Variables
# Decision Variable n
model.n = Var(model.Groups, domain=NonNegativeIntegers, bounds=(0,5))

# Density
model.rho = Var(bounds=(500,2000))

# Tb
model.Tb = Var(bounds=(200,1000))

#Tm
model.Tm = Var(bounds=(100,2000))

# RED
model.RED = Var(bounds=(0.001,1))

# Expressions
def delta_d_expr(m):
    return sum(m.n[g] * m.C_D[g] for g in m.Groups)
model.delta_d = Expression(rule=delta_d_expr)

def delta_p_expr(m):
    return sum(m.n[g] * m.C_P[g] for g in m.Groups)
model.delta_p = Expression(rule=delta_p_expr)

def delta_h_expr(m):
    return sum(m.n[g] * m.C_H[g] for g in m.Groups)
model.delta_h = Expression(rule=delta_h_expr)

def Ra_sq_expr(m):
    return (4 * (m.delta_d - m.delta_d_CO2)**2 + 
            (m.delta_p - m.delta_p_CO2)**2 + 
            (m.delta_h - m.delta_h_CO2)**2)
model.Ra_sq = Expression(rule=Ra_sq_expr)

def totalmw_expr(m):
    return sum(m.mws[g] * m.n[g] for g in m.Groups)
model.totalmw = Expression(rule=totalmw_expr)

def Vm_expr(m):
    return m.Vm0 + sum(m.n[g] * m.C_Vm[g] for g in m.Groups)
model.Vm = Expression(rule=Vm_expr)

def Cp_expr(m):
    return sum(m.n[g] * m.C_Cp[g] for g in m.Groups)
model.Cp = Expression(rule=Cp_expr)

# --- Constraints ---
# Boiling point lower bound
model.Tb_min_con = Constraint(expr=model.Tb >= 393)

# Melting point upper bound
model.Tm_min_con = Constraint(expr=model.Tm <= 313)

# RED upper bound
model.RED_limit = Constraint(expr=model.RED <= 1)

# Calculations for density, Tb, Tm and RED
# These were set as constraints such that variable bounds could be explicity defined
def rho_calc_rule(m):
    return m.rho == m.totalmw/(m.Vm0 + sum(m.n[g] * m.C_Vm[g] for g in m.Groups))
model.rho_con = Constraint(rule=rho_calc_rule)

def Tb_calc_rule(m):
    return m.Tb == m.Tb0 * log(sum(m.n[g] * m.C_Tb[g] for g in m.Groups) + 1e-6)
model.Tb_con = Constraint(rule=Tb_calc_rule)

def Tm_calc_rule(m):
    return m.Tm == m.Tm0 * log(sum(m.n[g] * m.C_Tm[g] for g in m.Groups) + 1e-6)
model.Tm_con = Constraint(rule=Tm_calc_rule)

def RED_calc_rule(m):
    return m.RED == sqrt(m.Ra_sq + 1e-8) / m.R0 
model.RED_con = Constraint(rule=RED_calc_rule)

# Structural Feasibility: Octet Rule
def valency_rule(m):
    total_groups = sum(m.n[g] for g in m.Groups)
    # Ensure solver define at least 1 group to avoid div by zero or negative logic issues
    return sum(m.n[g] * m.valency[g] for g in m.Groups) == 2 * (total_groups - 1)
model.valency_con = Constraint(rule=valency_rule)

# Structural Feasibility: Bonding Rule
def bonding_rule(m, j):
    total_groups = sum(m.n[g] for g in m.Groups)
    return m.n[j] * (m.valency[j] - 1) + 2 <= total_groups
model.bonding_con = Constraint(model.Groups, rule=bonding_rule)

# Objective function
def objective_rule(m):
    # RED: smaller is better
    RED_scaled = (m.RED - RED_min) / (RED_max - RED_min)
    # rho: larger i better -> negative sign to maximise
    rho_scaled = (m.rho - rho_min) / (rho_max - rho_min)
    # Cp: smaller is better
    Cp_scaled = (m.Cp - Cp_min) / (Cp_max - Cp_min)
    
    return( 
        m.w1 * RED_scaled      # minimise 
        - m.w2 * rho_scaled    # maximise rho via minus sign 
        + m.w3 * Cp_scaled     # minimise Cp
    )
model.obj = Objective(rule=objective_rule, sense=minimize)

solver = SolverFactory('gams') 
solver.options['solver'] = 'baron'

#%% CELL 3 - Generate Solution

# Call the solver, display performance metrics
results = solver.solve(model, tee=True)
print("\nSolution found!\n------ Molecule ------\n")

# Print all groups included in the final solution
tol = 1e-6
for g in model.Groups:
    x = value(model.n[g])
    if abs(x) < tol:
        x = 0
    x_rounded = round(x)
    if x_rounded > 0:
        print(g, x_rounded)

# Show Properties
print("\n------ Objectives ------\n")
print(f"Liquid Density: {value(model.rho):.2f} kg m-3")
print(f"Liquid Heat Capacity: {value(model.Cp):.2f} J K-1 mol-1 ")
print(f"RED: {value(model.RED):.3f}")
print(f"Molecular Weight: {value(model.totalmw):.2f} kg kmol-1")
print(f"Liquid Molar Volume: {value(model.Vm):.2f} m3 kmol-1")
print("\n------ Constraints ------\n")
print(f"Boiling point: {value(model.Tb):.2f} K")
print(f"Melting point: {value(model.Tm):.2f} K")
print("\n------ Additional ------\n")
print(f"Hansen parameters:")
print(f"  δd = {value(model.delta_d):.2f}")
print(f"  δp = {value(model.delta_p):.2f}")
print(f"  δh = {value(model.delta_h):.2f}")




