from pint import UnitRegistry
units = UnitRegistry()
import numpy as np
from math import ceil
import copy
import scipy.optimize as spo
from time import time

#Define functions first

def findMaxLength(kys):
    #used for the print formatting, not important
    ml = 0
    for k in list(kys):
        if len(k) > ml:
            ml = len(k)
    return ml

def computeRatedCurrent(d):
    # Produces rated current (amps) as a function of wire diameter (mm)
    resistance = 22.0/d**2.0  # ohms per 1000m
    ratedCurrent = 100.0/resistance**(0.72)
    return ratedCurrent

def computeResistance(d):
    # Produces resistance (ohms per 1000 m) for a given wire diameter (mm)
    resistance = 22.0/d**2.0  # ohms per 1000m
    return resistance

def diameter_from_dimension(wire_dimension,wire_type):
    # Used to convert between square and circle as required
    if wire_type.lower() == 'square':
        perm = wire_dimension*4
        return perm/np.pi
    elif wire_type.lower() == 'rectangle':
        perm = 2*(wire_dimension[0] + wire_dimension[1])
        return perm/np.pi
    else:
        return wire_dimension

def CW_res(f,*args):
    Re=args[0]
    D_h = args[1]
    epsilon=args[2]
    LHS = 1/(10**f)**0.5
    RHS = -2*np.log(epsilon/(3.7*D_h)+2.51/(Re*(10**f)**0.5))
    return (RHS-LHS)**2

def compute_motor_massProperties(dct):
    # Extremely simple mass build up

    # Unpack variables -------------------
    D_out          = dct['D_out']
    D_in           = dct['D_in']
    D_sh           = dct['D_sh']
    N_coils        = dct['N_coils']
    TPC            = dct['TPC']
    p              = dct['p']
    N_stators      = dct['N_stators']
    N_rotors       = dct['N_rotors']
    motor_type     = dct['motor_type']
    wire_dimension = dct['wire_dimension']
    wire_type      = dct['wire_type']
    if wire_type.lower() != 'rectangle':
        wire_thickness = dct['wire_thickness']
    m_1            = dct['m_1']
    B_mg           = dct['B_mg']
    alpha_i        = dct['alpha_i']
    V_1            = dct['V_1']
    s_1            = dct['s_1']
    N_1            = dct['N_1']
    N_AS           = dct['N_AS']
    I_1            = dct['I_1']
    n_s            = dct['n_s']
    s_coil = dct['s_coil']
    R_out = D_out/2.0
    R_in  = D_in/2.0
    R_sh  = D_sh/2.0

    # Define output dictionary ---------
    opt = {}

    # Densities -----------------------
    density_steel          = 8.05 * units.g / units.cm**3
    density_copper         = 8.69 * units.g / units.cm**3
    density_hdpe           = 0.97 * units.g / units.cm**3
    density_neodymium      = 7.50 * units.g / units.cm**3
    density_water          = 1.00 * units.g / units.cm**3
    density_ethyleneGlycol = 1.11 * units.g / units.cm**3

    coolantMixingFraction = 0.5
    density_coolant        = (1-coolantMixingFraction)*density_water + coolantMixingFraction*density_ethyleneGlycol

    # Compute mass of wire ---------------
    l = N_1 * m_1 * N_stators * s_coil
    if wire_type.lower() == 'square':
        outerArea = (wire_dimension)**2
        innerArea = (wire_dimension-2*wire_thickness)**2
        v = l*(outerArea - innerArea)
        v_inner = l * innerArea
        msc = v.to('m^3')*density_copper
        opt['wire'] = msc.to('g')
    elif wire_type.lower() == 'rectangle':
        xcArea = wire_dimension[0] * wire_dimension[1]
        v = l*xcArea
        msc = v.to('m^3')*density_copper
        opt['wire'] = msc.to('g')
    else:
        outerArea = (wire_dimension/2)**2*np.pi
        innerArea = (wire_dimension/2-wire_thickness)**2*np.pi
        v = l*(outerArea - innerArea)
        v_inner = l * innerArea
        msc = v.to('m^3')*density_copper
        opt['wire'] = msc.to('g')

    # Compute some geometry parameters ---------------
    if wire_type.lower() == 'rectangle':
        space_needed = max(wire_dimension.to('mm').magnitude) * units.mm
    else:
        space_needed = ceil(TPC**0.5) * wire_dimension
    slot_width = 2.5*space_needed  # variable name is a vestage, but is 2.5 times the width of a side of the wire bundle if it were square
                                   # Completely arbitrary selection

    # Define rotor and stator thicknesses, and shaft length ---------------
    if motor_type == 1:
        rotorThickness  = slot_width
    if motor_type == 2:
        rotorThickness  = dct['coreless_weight_penalty']*slot_width
    statorThickness = 1.5*slot_width
    shaftLength = (2 + N_rotors) * rotorThickness + N_stators * statorThickness

    # Compute mass of rotor and shaft -----------
    mass_rotor = np.pi * (R_out**2-R_in**2) * rotorThickness * density_steel
    mass_shaft = np.pi * R_sh**2 * shaftLength * density_steel
    opt['rotor'] = mass_rotor.to('g') * N_rotors
    opt['shaft'] = mass_shaft.to('g')

    # Compute mass of stator ----------------
    if motor_type == 1:
        mass_stator = np.pi * (R_out**2-R_in**2) * statorThickness * density_steel
    if motor_type == 2:
        mass_stator = np.pi * (R_out**2-R_in**2) * statorThickness * density_hdpe
    opt['stator'] = mass_stator.to('g')

    # Sum the relevant pieces and return ---------------
    opt['total'] = opt['rotor'] + opt['shaft'] + opt['wire'] + opt['stator']
    opt['rotating'] = opt['rotor'] + opt['shaft']
    return opt

def powerResidual(x, dct):
    '''
     This is the workhorse funciton of this analysis
     Balances shaft power with the power out from the electrical interactions
    '''

    # Checking to see which input is being solved for ---------------
    con1 = 'n_s' in list(dct.keys())
    con2 = 'I_1' in list(dct.keys())

    dct_in = copy.deepcopy(dct)
    if con1 and con2:
        con3 = dct['n_s'] != None
        con4 = dct['I_1'] != None
        if con3 and con4:
            raise ValueError('Problem is overdefined')
        if con3:
            I_1 = x[0] * units.ampere
            n_s = dct['n_s']
            dct_in['I_1'] = I_1
        if con4:
            I_1 = dct['I_1']
            n_s = x[0] * units.rpm
            dct_in['n_s'] = n_s
    elif con1:
        I_1 = x[0] * units.ampere
        n_s = dct['n_s']
        dct_in['I_1'] = I_1
    elif con2:
        I_1 = dct['I_1']
        n_s = x[0] * units.rpm
        dct_in['n_s'] = n_s
    else:
        raise ValueError('Processing Error')

    # Unpacking variables -----------
    I_a = I_1/2**0.5
    D_out          = dct['D_out']
    D_in           = dct['D_in']
    D_sh           = dct['D_sh']
    N_coils        = dct['N_coils']
    TPC            = dct['TPC']
    p              = dct['p']
    N_stators      = dct['N_stators']
    N_rotors       = dct['N_rotors']
    motor_type     = dct['motor_type']
    wire_dimension = dct['wire_dimension']
    wire_type      = dct['wire_type']
    if wire_type.lower() != 'rectangle':
        wire_thickness = dct['wire_thickness']
    m_1            = dct['m_1']
    B_mg           = dct['B_mg']
    alpha_i        = dct['alpha_i']
    V_1            = dct['V_1']
    frequency      = dct['frequency']

    # Checks if mode is assigned, if not assume mode 0 -----------
    if 'mode' in list(dct.keys()):
        mode          = dct['mode']
    else:
        mode = 0

    # Computes some necessary geometry quantities --------------
    R_out = D_out/2.0
    R_in  = D_in/2.0

    if motor_type == 1:
        s_1           = N_coils   # Number of slots, must be >2*p
    if motor_type == 2:
        s_1           = 2*N_coils   # Number of sides, must be >2*p

    if N_stators == N_rotors or N_stators+1 == N_rotors or N_stators-1 == N_rotors:
        if N_stators == N_rotors:
            N_AS = 2*N_stators - 1
        if N_stators < N_rotors:
            N_AS = 2*N_stators
        if N_stators > N_rotors:
            N_AS = 2*N_stators - 2
    else:
        raise ValueError('Invalid rotor-stator configuration')

    N_1 = TPC*(N_coils/m_1)   # Number of turns per phase (series) per stator
    # =====================================
    # Power Out 1
    # =====================================
    '''
    This short section is essentially P=T*omega (the useful shaft power out of the motor)
    Torque constant (k_tau) is computed from equations in Gieras
    Note that either I_a or n_s contains a guess value at this point, we must still solve the residual
    '''

    # Stuff from Gieras  --------
    tau = np.pi/p
    w_c = 2*np.pi/s_1
    beta = w_c/tau
    q_1  = s_1/(2*p*m_1)
    k_p1 = np.sin(beta * np.pi/2)
    k_d1 = (       np.sin(np.pi/(2*m_1     )) /
            (q_1 * np.sin(np.pi/(2*m_1*q_1)))    )
    k_w1 = k_d1 * k_p1
    k_tau = np.pi/(2*2**0.5) * alpha_i * k_w1 * m_1 * N_1 * B_mg * (R_out**2 - R_in**2) * N_AS

    # T = I * k_tau (by definition for any motor)
    Torque = k_tau * I_a

    #  P = T * omega
    P_out1 = k_tau * I_a * n_s
    # =====================================
    # Power Out 2
    # =====================================
    '''
    This section takes in the currently guessed value of either I_a or n_s and computes the losses in the motor
    Then, the P_out = P_in - P_loss
    This must balance with the shaft output, otherwise we are not at a real physical operating point
    Details will be sparse as it is mostly an implementation of the method in the Gieras book
    '''

    Delta_P = 0.0 * units.W  # Total P_loss
    powerDict = {}           # A useful output with breakdown of all powers

    R_out = D_out/2.0
    R_in  = D_in/2.0
    R_sh  = D_sh/2.0

    P_in = V_1 * I_1
    I_a = I_1/2**(0.5)      # current in one winding (phase)

    # Compute the approximate length of one turn of coil
    Delta_R = R_out - R_in
    outerArcLength = 2*np.pi/s_1 * (R_out)
    innerArcLength = 2*np.pi/s_1 * (R_in)
    if motor_type == 1:
        s_coil = 2*Delta_R + outerArcLength + innerArcLength
    if motor_type == 2:
        s_coil = 2*Delta_R + 3*outerArcLength + 3*innerArcLength

    dct_in['s_1']  = s_1
    dct_in['N_1']  = N_1
    dct_in['N_AS'] = N_AS
    dct_in['s_coil'] = s_coil
    # Return motor mass properties
    mso = compute_motor_massProperties(dct_in)
    # =============
    # DC Losses
    # =============
    '''
    These are the I^2 * R losses in the copper
    Generally significant, but not driving
    '''
    effectiveWireDiameter = diameter_from_dimension(wire_dimension,wire_type)
    ratedCurrent = computeRatedCurrent(effectiveWireDiameter.to('mm').magnitude)
#     wireResistance = computeResistance(effectiveWireDiameter.to('mm').magnitude) * units.ohms/units.m /1000.
    conductivity = 50e6*units.siemens/units.m
    con_rho = 1/conductivity
    if wire_type.lower() == 'square':
        wireArea = (wire_dimension)**2
    elif wire_type.lower() == 'rectangle':
        wireArea = wire_dimension[0] * wire_dimension[1]
    else:
        wireArea = (wire_dimension/2)**2*np.pi
    wireResistance = con_rho/wireArea

    lengthWireSinglePhase = N_1 * N_stators * s_coil
    if I_a <= ratedCurrent*units.ampere:
        powerDict['Electrical Resistance'] = 2.5 * lengthWireSinglePhase * wireResistance * I_a**2
    else:
        powerDict['Electrical Resistance'] = float('nan') * units.W

    # =============
    # Stator Core
    # =============
    '''
    Losses due to eddy currents in the stator (cored motors only, zero in coreless)
    The method in Gieras has been simplified here in order to not require FEA
    Makes some conservative assumptions
    Should generally be small, so long as lamination thickness is reasonable
    This model in general should not be run (indicated by setting the 'run_losses' input to False)
    '''
    if dct['run_losses']:
        if motor_type == 1:
            density_steel          = 8.05 * units.g / units.cm**3

            sigma_fe = 2e6 *units.siemens/units.m
            d_fe = 1.0*units.mm  # Lamination thickness, should be small
            m_fe = mso['stator']
            Bmx1 = B_mg
            Bmz1 = B_mg
            eta_d = 1.2

            Pefe = np.pi**2/6 * sigma_fe / density_steel * frequency**2 *d_fe**2 * m_fe * (Bmx1**2 + Bmz1**2) * eta_d**2
            Pefe = Pefe.to('W')

            rct = 3.0*units.m**4/units.Hz/units.kg
            Phfe = rct * frequency/100 * m_fe * (Bmx1**2 + Bmz1**2) * eta_d**2
            Phfe = Pefe.to('W')

            powerDict['Stator Core'] = Pefe + Phfe

        if motor_type==2:
            powerDict['Stator Core'] = 0.0*units.W

    # =============
    # Permanent Magnets
    # =============
    '''
    Losses due to eddy currents and heating in the rotor magnets
    (cored motors only, negligible in coreless).
    The method in Gieras has been simplified here in order to not require FEA
    Makes some conservative assumptions.
    Should generally be small
    This model in general should not be run (indicated by setting the 'run_losses' input to False)
    '''
    if dct['run_losses']:
        if motor_type == 1:
            if wire_type.lower() == 'rectangle':
                WD = max(wire_dimension.to('mm').magnitude) * units.mm
            else:
                WD = wire_dimension
            space_needed = ceil(TPC**0.5) * WD
            magnetThickness = 2.5*space_needed
            sigma_pm = 6.67e6 *units.siemens/units.m
            mu_0 = 1.25663706e-6 *units.m*units.kg/units.s**2/units.ampere**2
            beta  = np.pi/(tau * R_in)
            omega_v = beta * np.pi * D_out * n_s
            k_z = 1+outerArcLength/(D_out-D_in)
            alpha = (omega_v * mu_0*sigma_pm)**(0.5)
            Gma = 2*WD/(np.pi*0.5*(D_out+D_in)/s_1)
            asl = 4/np.pi * (0.5 + Gma**2/(0.78-2*Gma**2)) * np.sin(1.6*np.pi*Gma)
            kpa = 2*WD/(magnetThickness + 1.0*units.mm)
            kC = 1.05
            btasl = 0.5*(1-1/(1+kpa**2)**0.5)
            B_sl = B_mg*alpha_i * asl*btasl*kC
            mu_rrec = 1.0
            k = alpha/2**(0.5)
            un = (beta**4/k**4).to_base_units().units
            a_Rv = 1/2**(0.5) * ((4*un+beta**4/k**4)**0.5 + beta**2/k**2)**0.5
            S_pm = N_AS * alpha_i * np.pi/4 * (D_out**2 - D_in**2)

            Ppm = 0.5 * a_Rv * k_z * alpha**2/beta**2 * (B_sl/(mu_0*mu_rrec))**2 * k/sigma_pm * S_pm
            Ppm = Ppm.to('W')
            powerDict['Permanent Magnets'] = Ppm
        if motor_type == 2:
            powerDict['Permanent Magnets'] = 0.0*units.W

    # =============
    # Rotor Core
    # =============
    '''
    Losses due to eddy currents and heating in the rotor core or backplate
    (cored motors only, negligible in coreless)
    The method in Gieras has been simplified here in order to not require FEA
    Makes some conservative assumptions
    Should generally be small
    This model in general should not be run (indicated by setting the 'run_losses' input to False)
    '''
    if dct['run_losses']:
        if motor_type == 1:
            dv = beta**2/k**2
            aR = 1.45 * dv.to_base_units().units
            aX = 0.85 * dv.to_base_units().units

            tm1 = 4 * aR**2 * aX**2
            tm2 = aR**2 - aX**2 + dv
            tm3 = -aR**2 + aX**2 - dv

            a_Rfe = 1/2**0.5 * ((tm1 + tm2**2)**0.5 + tm3)**0.5

            Prd = 0.5 * a_Rfe * k_z * alpha**2/beta**2 * (B_sl/(mu_0*mu_rrec))**2 * k/sigma_fe * S_pm
            Prd = Prd.to('W')
            powerDict['Rotor Core'] = Prd
        if motor_type == 2:
            powerDict['Rotor Core'] = 0.0*units.W

    # =============
    # Bearing Friction
    # =============
    '''
    Losses due to bearing friction
    k_fb is between 1 and 3
    Should generally significant
    '''
    k_fb = 2*units.m**2/units.s**2
    powerDict['Friction'] = 0.06 * k_fb * n_s.to('revolution/s').magnitude/units.sec * mso['rotating']

    # =============
    # Air Friction
    # =============
    '''
    Losses due to drag of the disks
    Assumes air as working medium
    Generally pretty substantial, especially for large disks or high RPM
    '''
    rho = 1.23*units.kg/units.m**3
    mu = 17.1e-6 * units.N * units.s / units.m**2
    Re = rho * R_out**2 * 2 * np.pi * n_s / mu
    c_f = 3.87/Re**(0.5)
    powerDict['Wind'] = 0.5 * c_f * rho * (n_s)**3 * (R_out**5 - R_in**5) * N_rotors

    # =============
    # Unmodeled Knock Down
    # =============
    '''
    Applies a 20% flat penalty if the loss functions are skipped
    Generally a better estimate than actually running the functions, which can be extremely fragile
    '''
    if not dct['run_losses']:
        if motor_type == 1:
            lossesSoFar = 0.0*units.W
            for k, vl in powerDict.items():
                powerDict[k] = powerDict[k].to('W')
                lossesSoFar += vl
            powerDict['Unmodeled'] = 0.20*lossesSoFar
    # =============
    # Cooling
    # =============
    '''
    Approximates cooling requirements, and power required to pump fluid through the motor
    Cooling is a simple P=mdot*c*delta_T
    Coolant is assumed to heat up 20 deg C (delta_T variable name).   This is perhaps a bit agressive
    Takes out all lost power up to this point
    Pressure loss computed with simple turbulent flow relation
    Generally not large, unless diameter of the pipe is driven too low
    '''
    density_water          = 1.00 * units.g / units.cm**3
    density_ethyleneGlycol = 1.11 * units.g / units.cm**3
    coolantMixingFraction  = 0.5
    density_coolant        = (1-coolantMixingFraction)*density_water + coolantMixingFraction*density_ethyleneGlycol

    c_eg = 3140*units.J/units.kg / units.delta_degC
    c_water = 4200*units.J/units.kg / units.delta_degC
    c_mix = (1-coolantMixingFraction)*c_water + coolantMixingFraction*c_eg
#     c_mix = 100*units.J/units.kg / units.delta_degC

    delta_T = 20*units.delta_degC
    g = 9.8*units.m/units.s**2
    nu_water = 0.5531e-6 * units.m**2 / units.s

    P_cooled = 0.0*units.W
    for k, vl in powerDict.items():
        powerDict[k] = powerDict[k].to('W')
        P_cooled += vl

    if wire_type.lower() == 'rectangle':
        singleChannelArea = 0.5 * wire_dimension[0] * wire_dimension[1]
        singleChannelPerimeter = 2*(wire_dimension[0] + wire_dimension[1])
        hydraulicDiameter = 4*singleChannelArea/singleChannelPerimeter
        N_channels = TPC-1
        totalArea = singleChannelArea * N_channels

        roughness = dct['print_roughness']
        V_coolant = P_cooled/(3 * density_coolant * totalArea * c_mix * delta_T)
        Re = V_coolant * hydraulicDiameter/nu_water
        res = spo.minimize(CW_res,-2,args=(Re.to('').magnitude,
                                             hydraulicDiameter.to('m').magnitude,
                                             roughness.to('m').magnitude))
        f_d = 10**(res['x'][0])
        del_p = f_d * density_coolant /2 * V_coolant**2/hydraulicDiameter * lengthWireSinglePhase

        powerDict['Cooling Flow'] = 3*V_coolant*totalArea*del_p

    else:
        if wire_type.lower() == 'square':
            innerArea = (wire_dimension-2*wire_thickness)**2
        else:
            innerArea = (wire_dimension/2-wire_thickness)**2*np.pi

        V_coolant = P_cooled/(3 * density_coolant * innerArea * c_mix * delta_T)

        D_coolant = wire_dimension-2*wire_thickness
        S = 64.0*nu_water/(2*g) * V_coolant/D_coolant**2
        del_p = S*density_coolant*g*lengthWireSinglePhase

        powerDict['Cooling Flow'] = 3*V_coolant*innerArea*del_p

    # =============
    # Sum and Process Powers
    # =============
    for k, vl in powerDict.items():
        powerDict[k] = powerDict[k].to('W')
        Delta_P += vl

    P_out2 = P_in - Delta_P
    # =====================================
    # Post Process
    # =====================================
    '''
    Mode 0 is standard residual solving mode
    Modes 1 and 2 are debug modes
    Mode 3 is the return mode which returns all the outputs for viewing (assumes a valid combination of I_1 and n_s)
    '''
    if mode==0:
        return (P_out1.to('W').magnitude - P_out2.to('W').magnitude)**2
    if mode == 1:
        return P_out1.to('W').magnitude
    if mode == 2:
        return P_out2.to('W').magnitude
    if mode == 3:
        eta = P_out1.to('W')/(V_1*I_1)

        SF = 2.0
        tauMax_steel = 150*units.MPa
        T = (I_1*V_1/(n_s)).to('N*m')
        Dmin_sh = 2*(2/np.pi * T/(tauMax_steel/SF))**(1/3.0)

        output = {}
        output['P_shaft'] = P_out1.to('kW')
        output['P_loss']  = Delta_P.to('W')
        output['P_losses'] = powerDict
        output['Torque']   = Torque.to('N*m')
        output['Torque Constant']   = k_tau.to('m^2*T')
        output['Rotational Speed'] = n_s.to('rpm')
        output['Line Current'] = I_1.to('ampere')
        output['Efficiency'] = eta.to('').magnitude
        output['Shaft Feasible'] = D_sh >= Dmin_sh
        output['Mass'] = mso['total']
        output['Masses'] = mso
        msp = output['P_shaft'] / output['Mass']
        output['Mass Specific Power'] = msp.to('W/kg')
        dsp = output['P_shaft'] / D_out
        output['Diameter Specific Power'] = dsp.to('W/m')
        output['Coolant Velocity'] = V_coolant.to('m/s')
        output['Coolant Pressure Drop'] = del_p.to('kPa')
        output['Pipe Reynolds Number'] = Re.to('').magnitude
        output['Darcy Friction Factor'] = f_d
        return output

def check_config(dct):
    """ Checks for a valid motor configuration. """
    error = ValueError('Invalid Motor Configuration')
    if dct['motor_type'] == 1:
        if dct['N_coils'] <= 2 * dct['p']:
            raise error
    if dct['motor_type'] == 2:
        if dct['N_coils'] <= dct['p']:
            raise error
    if abs(dct['N_rotors'] - dct['N_stators']) >= 2:
        raise error
    if dct['N_coils']%dct['m_1'] != 0:
        raise error
    if dct['I_1'].to('ampere').magnitude > dct['I_max'].to('ampere').magnitude:
        raise error
    pass

def simulate_motor(dct, tol=1e-6, verbosity=0):
    """ Simulates motor for a given configuration and current level. """
    t1 = time()
    check_config(dct)
    # Run (solve for whichever value you did not specify above) =======================
    res = spo.minimize(powerResidual, 1.0, args=(dct), method='SLSQP')
    res['solve_time'] = (time()-t1)*units.s
    if res['message'] != 'Optimization terminated successfully.' or abs(res['fun']) > tol:
        print(res)
        raise RuntimeError('Solver did not converge')
    if verbosity > 0:
        print_results(res, dct)
    return res

def print_results(res, dct):
    """ Prints the results for the simulated motor. """
    dct_post = copy.deepcopy(dct)
    dct_post['mode']=3
    opt = powerResidual([res['x'][0]],dct_post)
    cw = findMaxLength(opt.keys())
    pstr = ''
    for k,v in opt.items():
        if k!='P_losses' and k!='Masses':
            pstr += '%s : %s\n'%(k.ljust(cw,' '),v)
        if k == 'P_loss':
            kwd = 'P_losses'
            cw2 = findMaxLength(opt[kwd].keys())
            for k2, v2 in opt[kwd].items():
                pstr += '    %s : %s\n'%(k2.ljust(cw2,' '),v2)
        if k == 'Mass':
            kwd = 'Masses'
            cw2 = findMaxLength(opt[kwd].keys())
            for k2, v2 in opt[kwd].items():
                if k2 != 'total' and k2 != 'rotating':
                    pstr += '    %s : %s\n'%(k2.ljust(cw2,' '),v2)
    print(pstr)

def baseline():
    """ Returns baseline inputs to motor model that closes. """
    dct = {}
    # Variables =======================
    dct['D_out']          = 13*units.cm    # Outer Diameter of the motor
    dct['D_in']           = 7.6*units.cm   # Inner Diameter of the motor
    dct['D_sh']           = 1.0*units.cm   # Diameter of motor shaft
    dct['N_coils']        = 18             # Number of coils on each stator, must be =n*m_1--If motor_type==1 must be >2*p, If motor_type==2 must be >p,
    dct['TPC']            = 10             # Number of turns per coil on each stator
    dct['p']              = 16             # Half the number of poles on each rotor
    dct['N_stators']      = 1              # Number of Stators
    dct['N_rotors']       = 2              # Number of Rotors
    dct['motor_type']     = 2              # 1: feromagnetic core with non-overlapping coils, 2: coreless with overlapping coils
    dct['wire_dimension'] = [0.15,3.0]*units.mm   # Reference dimension of the wire: Diameter for circle, width for square
    dct['wire_type']      = 'rectangle'       # Wire geometry, either circle or square
    # dct['wire_dimension'] = 2*units.mm   # Reference dimension of the wire: Diameter for circle, width for square
    # dct['wire_thickness'] = 0.4 * units.mm # Thickness of copper in the wire, interior becomes coolant path
    # dct['wire_type']      = 'square'       # Wire geometry, either circle or square
    # Constants =======================
    dct['m_1']            = 3                 # Number of phases, assumed to be the standard 3
    dct['B_mg']           = 0.65 * units.T    # Peak magnetic field flux density (pretty constant)
    dct['alpha_i']        = 2.0/np.pi         # A constant used in Gieras
    dct['V_1']            = 380*units.volts   # Line Voltage
    dct['coreless_weight_penalty'] = 1.5      # Penalty for having additional magnets
    dct['frequency']  = 50*units.Hz           # AC frequency
    dct['run_losses'] = False                 # Flag whether or not to run the loss functions (cored motor only).  Model not validated, should leave false
    dct['print_roughness'] = 0.05*units.mm
    # Compute max current the motor can take =======================
    dct['equivalent_diameter'] = diameter_from_dimension(dct['wire_dimension'],dct['wire_type'])
    dct['I_max']     = computeRatedCurrent(dct['equivalent_diameter'].to('mm').magnitude) * units.ampere
    # Values to be solved =======================
    dct['n_s']            = None #8000*units.rpm
    dct['I_1']            = dct['I_max']
    return dct

if __name__ == '__main__':
    dct = baseline()
    res = simulate_motor(dct, verbosity=1)




