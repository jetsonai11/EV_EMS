"""
##############################################################################
##############################################################################
Pythonic copy of NREL's FASTSim
(Future Automotive Systems Technology Simulator)
Input Arguments
1) cyc: dictionary defining drive cycle to be simulated
        cyc['cycSecs']: drive cycle time in seconds (begins at zero)
        cyc['cycMps']: desired vehicle speed in meters per second
        cyc['cycGrade']: road grade
        cyc['cycRoadType']: Functional Class of GPS data
2) veh: dictionary defining vehicle parameters for simulation
Output Arguments
A dictionary containing scalar values summarizing simulation
(mpg, Wh/mi, avg engine power, etc) and/or time-series results for component
power and efficiency. Currently these values are user clarified in the code by assigning values to the 'output' dictionary.
    List of Abbreviations
    cur = current time step
    prev = previous time step

    cyc = drive cycle
    secs = seconds
    mps = meters per second
    mph = miles per hour
    kw = kilowatts, unit of power
    kwh = kilowatt-hour, unit of energy
    kg = kilograms, unit of mass
    max = maximum
    min = minimum
    avg = average
    fs = fuel storage (eg. gasoline/diesel tank, pressurized hydrogen tank)
    fc = fuel converter (eg. internal combustion engine, fuel cell)
    mc = electric motor/generator and controller
    ess = energy storage system (eg. high voltage traction battery)

    chg = charging of a component
    dis = discharging of a component
    lim = limit of a component
    regen = associated with regenerative braking
    des = desired value
    ach = achieved value
    in = component input
    out = component output

##############################################################################
##############################################################################
"""

### Import necessary python modules
import numpy as np
import warnings
import csv
import BX_lib
import gym
from gym import spaces
warnings.simplefilter('ignore')

def get_standard_cycle(cycle_name):
    csv_path = '..//cycles//'+cycle_name+'.csv'
    data = dict()
    dkeys=[]
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(data)==0: # initialize all elements in dictionary based on header
                for ii in range(len(row)):
                    data[row[ii]] = []
                    dkeys.append( row[ii] )
            else: # append values
                for ii in range(len(row)):
                    try:
                        data[dkeys[ii]].append( float(row[ii]) )
                    except:
                        data[dkeys[ii]].append( np.nan )
    for ii in range(len(dkeys)):
        data[dkeys[ii]] = np.array(data[dkeys[ii]])
    return data

def get_veh(vnum):
    with open('..//docs//FASTSim_py_veh_db.csv','r') as csvfile:

        reader = csv.reader(csvfile)
        vd = dict()
        data = dict()
        z=0

        for i in reader:
            data[z]=i
            z=z+1

        variables = data[0]
        del data[0] # deletes the first list, which corresponds to the header w/ variable names
        vd=data

        ### selects specified vnum from vd
        veh = dict()
        variables = ['selection','name', 'vehPtType', 'dragCoef', 'frontalAreaM2', 'gliderKg', 'vehCgM', 'driveAxleWeightFrac', 'wheelBaseM', 'cargoKg', 'vehOverrideKg', 'maxFuelStorKw', 'fuelStorSecsToPeakPwr', 'fuelStorKwh', 'fuelStorKwhPerKg', 'maxFuelConvKw', 'fcEffType', 'fcAbsEffImpr', 'fuelConvSecsToPeakPwr', 'fuelConvBaseKg', 'fuelConvKwPerKg', 'maxMotorKw', 'motorPeakEff', 'motorSecsToPeakPwr', 'mcPeKgPerKw', 'mcPeBaseKg', 'maxEssKw', 'maxEssKwh', 'essKgPerKwh', 'essBaseKg', 'essRoundTripEff', 'essLifeCoefA', 'essLifeCoefB', 'wheelInertiaKgM2', 'numWheels', 'wheelRrCoef', 'wheelRadiusM', 'wheelCoefOfFric', 'minSoc', 'maxSoc', 'essDischgToFcMaxEffPerc', 'essChgToFcMaxEffPerc', 'maxAccelBufferMph', 'maxAccelBufferPercOfUseableSoc', 'percHighAccBuf', 'mphFcOn', 'kwDemandFcOn', 'altEff', 'chgEff', 'auxKw', 'forceAuxOnFC', 'transKg', 'transEff', 'compMassMultiplier', 'essToFuelOkError', 'maxRegen', 'valUddsMpgge', 'valHwyMpgge', 'valCombMpgge', 'valUddsKwhPerMile', 'valHwyKwhPerMile', 'valCombKwhPerMile', 'valCdRangeMi', 'valConst65MphKwhPerMile', 'valConst60MphKwhPerMile', 'valConst55MphKwhPerMile', 'valConst45MphKwhPerMile', 'valUnadjUddsKwhPerMile', 'valUnadjHwyKwhPerMile', 'val0To60Mph', 'valEssLifeMiles', 'valRangeMiles', 'valVehBaseCost', 'valMsrp', 'minFcTimeOn', 'idleFcKw','fuelKwhPerKg']
        if vnum in vd:
            for i in range(len(variables)):
                vd[vnum][i]=str(vd[vnum][i])
                if vd[vnum][i].find('%') != -1:
                    vd[vnum][i]=vd[vnum][i].replace('%','')
                    vd[vnum][i]=float(vd[vnum][i])
                    vd[vnum][i]=vd[vnum][i]/100.0
                elif vd[vnum][i].find('TRUE') != -1 or vd[vnum][i].find('True') != -1 or vd[vnum][i].find('true') != -1:
                    vd[vnum][i]=1
                elif vd[vnum][i].find('FALSE') != -1 or vd[vnum][i].find('False') != -1 or vd[vnum][i].find('false') != -1:
                    vd[vnum][i]=1
                else:
                    try:
                        vd[vnum][i]=float(vd[vnum][i])
                    except:
                        pass
                veh[variables[i]]=vd[vnum][i]

    ######################################################################
    ### Append additional parameters to veh structure from calculation ###
    ######################################################################

    ### Build roadway power lookup table
    veh['MaxRoadwayChgKw_Roadway'] = range(6)
    veh['MaxRoadwayChgKw'] = [0]*len(veh['MaxRoadwayChgKw_Roadway'])
    veh['chargingOn'] = 0

     # Checking if a vehicle has any hybrid components
    if veh['maxEssKwh']==0 or veh['maxEssKw']==0 or veh['maxMotorKw']==0:
        veh['noElecSys'] = 'TRUE'

    else:
        veh['noElecSys'] = 'FALSE'

    # Checking if aux loads go through an alternator
    if veh['noElecSys']=='TRUE' or veh['maxMotorKw']<=veh['auxKw'] or veh['forceAuxOnFC']=='TRUE':
        veh['noElecAux'] = 'TRUE'

    else:
        veh['noElecAux'] = 'FALSE'

    veh['vehTypeSelection'] = np.copy( veh['vehPtType'] ) # Copying vehPtType to additional key

    ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
    ### see "FC Model" tab in FASTSim for Excel

    if veh['maxFuelConvKw']>0:


        # Discrete power out percentages for assigning FC efficiencies
        fcPwrOutPerc = np.array([0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00])

        # Efficiencies at different power out percentages by FC type
        eff_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
        eff_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
        eff_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
        eff_fuel_cell = np.array([0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
        eff_hd_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])


        if veh['fcEffType']==1:
            eff = np.copy( eff_si ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==2:
            eff = np.copy( eff_atk ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==3:
            eff = np.copy( eff_diesel ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==4:
            eff = np.copy( eff_fuel_cell ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==5:
            eff = np.copy( eff_hd_diesel ) + veh['fcAbsEffImpr']

        inputKwOutArray = fcPwrOutPerc * veh['maxFuelConvKw']
        fcPercOutArray = np.r_[np.arange(0,3.0,0.1),np.arange(3.0,7.0,0.5),np.arange(7.0,60.0,1.0),np.arange(60.0,105.0,5.0)] / 100
        fcKwOutArray = veh['maxFuelConvKw'] * fcPercOutArray
        fcEffArray = np.array([0.0]*len(fcPercOutArray))

        for j in range(0,len(fcPercOutArray)-1):

            low_index = np.argmax(inputKwOutArray>=fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = eff[low_index-1]
            fcinterp_y_2 = eff[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        fcEffArray[-1] = eff[-1]
        veh['fcEffArray'] = np.copy(fcEffArray)
        veh['fcKwOutArray'] = np.copy(fcKwOutArray)
        veh['maxFcEffKw'] = np.copy(veh['fcKwOutArray'][np.argmax(fcEffArray)])
        veh['fcMaxOutkW'] = np.copy(max(inputKwOutArray))
        veh['minFcTimeOn'] = 30

    else:
        veh['fcKwOutArray'] = np.array([0]*101)
        veh['maxFcEffKw'] = 0
        veh['fcMaxOutkW'] = 0
        veh['minFcTimeOn'] = 30

    ### Defining MC efficiency curve as lookup table for %power_in vs power_out
    ### see "Motor" tab in FASTSim for Excel
    if veh['maxMotorKw']>0:

        maxMotorKw = veh['maxMotorKw']

        mcPwrOutPerc = np.array([0.00, 0.02, 0.04, 0.06, 0.08,	0.10,	0.20,	0.40,	0.60,	0.80,	1.00])
        large_baseline_eff = np.array([0.83, 0.85,	0.87,	0.89,	0.90,	0.91,	0.93,	0.94,	0.94,	0.93,	0.92])
        small_baseline_eff = np.array([0.12,	0.16,	 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93,	0.93,	0.92])

        modern_max = 0.95
        modern_diff = modern_max - max(large_baseline_eff)

        large_baseline_eff_adj = large_baseline_eff + modern_diff

        mcKwAdjPerc = max(0.0,min((maxMotorKw - 7.5)/(75.0-7.5),1.0))
        mcEffArray = np.array([0.0]*len(mcPwrOutPerc))

        for k in range(0,len(mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc*large_baseline_eff_adj[k] + (1-mcKwAdjPerc)*(small_baseline_eff[k])

        mcInputKwOutArray = mcPwrOutPerc * maxMotorKw

        mcPercOutArray = np.linspace(0,1,101)
        mcKwOutArray = np.linspace(0,1,101) * maxMotorKw

        mcFullEffArray = np.array([0.0]*len(mcPercOutArray))

        for m in range(1,len(mcPercOutArray)-1):
            low_index = np.argmax(mcInputKwOutArray>=mcKwOutArray[m])

            fcinterp_x_1 = mcInputKwOutArray[low_index-1]
            fcinterp_x_2 = mcInputKwOutArray[low_index]
            fcinterp_y_1 = mcEffArray[low_index-1]
            fcinterp_y_2 = mcEffArray[low_index]

            mcFullEffArray[m] = (mcKwOutArray[m] - fcinterp_x_1)/(fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        mcFullEffArray[0] = mcFullEffArray[1]
        mcFullEffArray[-1] = mcEffArray[-1]

        mcKwInArray = mcKwOutArray / mcFullEffArray
        mcKwInArray[0] = 0

        veh['mcKwInArray'] = np.copy(mcKwInArray)
        veh['mcKwOutArray'] = np.copy(mcKwOutArray)
        veh['mcMaxElecInKw'] = np.copy(max(mcKwInArray))
        veh['mcFullEffArray'] = np.copy(mcFullEffArray)
        veh['mcEffArray'] = np.copy(mcEffArray)

    else:
        veh['mcKwInArray'] = np.array([0.0] * 101)
        veh['mcKwOutArray'] = np.array([0.0]* 101)
        veh['mcMaxElecInKw'] = 0

    veh['mcMaxElecInKw'] = max(veh['mcKwInArray'])

    ### Specify shape of mc regen efficiency curve
    ### see "Regen" tab in FASTSim for Excel
    veh['regenA'] = 500.0
    veh['regenB'] = 0.99

    ### Calculate total vehicle mass
    if veh['vehOverrideKg']==0 or veh['vehOverrideKg']=="":
        if veh['maxEssKwh']==0 or veh['maxEssKw']==0:
            ess_mass_kg = 0.0
        else:
            ess_mass_kg = ((veh['maxEssKwh']*veh['essKgPerKwh'])+veh['essBaseKg'])*veh['compMassMultiplier']
        if veh['maxMotorKw']==0:
            mc_mass_kg = 0.0
        else:
            mc_mass_kg = (veh['mcPeBaseKg']+(veh['mcPeKgPerKw']*veh['maxMotorKw']))*veh['compMassMultiplier']
        if veh['maxFuelConvKw']==0:
            fc_mass_kg = 0.0
        else:
            fc_mass_kg = (((1/veh['fuelConvKwPerKg'])*veh['maxFuelConvKw']+veh['fuelConvBaseKg']))*veh['compMassMultiplier']
        if veh['maxFuelStorKw']==0:
            fs_mass_kg = 0.0
        else:
            fs_mass_kg = ((1/veh['fuelStorKwhPerKg'])*veh['fuelStorKwh'])*veh['compMassMultiplier']
        veh['vehKg'] = veh['cargoKg'] + veh['gliderKg'] + veh['transKg']*veh['compMassMultiplier'] + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg
    else:
        veh['vehKg'] = np.copy( veh['vehOverrideKg'] )

    return veh

def sim_drive( cyc , veh ):

    if veh['vehPtType']==1:

        # If no EV / Hybrid components, no SOC considerations.

        initSoc = 0.0
        output = sim_drive_sub( cyc , veh , initSoc )

    elif veh['vehPtType']==2:

        #####################################
        ### Charge Balancing Vehicle SOC ###
        #####################################

        # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
        # Iterating until tolerance met or 30 attempts made.
        ##########################################################################
        ##########################################################################
        ##########################################################################
        initSoc = 0.6
#         initSoc = (veh['maxSoc']+veh['minSoc'])/2.0
#         print('004/',initSoc)
        ess2fuelKwh = 1.0
        sim_count = 0
        ##########################################################################
        ##########################################################################
        ##########################################################################
        cyc_num = 'one'
#         cyc_num = 'mul'
        if cyc_num=='one':
            # one cycle
            sim_count += 1
            output = sim_drive_sub( cyc , veh , initSoc )
            ess2fuelKwh = abs( output['ess2fuelKwh'] )
        else:
        ##########################################################################
        ##########################################################################
        ##########################################################################
            # multiple cycle
            while ess2fuelKwh>veh['essToFuelOkError'] and sim_count<30:
                sim_count += 1
                output = sim_drive_sub( cyc , veh , initSoc )
                ess2fuelKwh = abs( output['ess2fuelKwh'] )
                initSoc = min(1.0,max(0.0,output['final_soc']))
        np.copy( veh['maxSoc'] )
        output = sim_drive_sub( cyc , veh , initSoc )

    elif veh['vehPtType']==3 or veh['vehPtType']==4:

        # If EV, initializing initial SOC to maximum SOC.

        initSoc = np.copy( veh['maxSoc'] )
        output = sim_drive_sub( cyc , veh , initSoc )

    return output




class FASTSimEnvironment(gym.Env):
    def __init__(self):
        self.min_power=-30
        self.max_power=30     # P_dmd [Kw]
        self.min_vel=0
        self.max_vel=30       # v [mps]
        #self.min_soc=0.4
        #self.max_soc=0.9     # soc
        self.min_action=0
        self.max_action=1     # P split ratio

        self.low_state = np.array(
            [self.min_power, self.min_vel], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_power, self.max_vel], dtype=np.float32
        )

        self.state_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        ############################
        ###   Define Constants   ###
        ############################

        self.veh = getveh(24)
        self.cyc = FASTSim_RL.get_standard_cycle("UDDS")
        self.initSoc = 0.6

        self.airDensityKgPerM3 = 1.2 # Sea level air density at approximately 20C
        self.gravityMPerSec2 = 9.81
        self.mphPerMps = 2.2369
        self.kWhPerGGE = 33.7
        self.metersPerMile = 1609.00
        self.maxTracMps2 = ((((self.veh['wheelCoefOfFric']*self.veh['driveAxleWeightFrac']*self.veh['vehKg']*self.gravityMPerSec2)/(1+((self.veh['vehCgM']*self.veh['wheelCoefOfFric'])/self.veh['wheelBaseM']))))/(self.veh['vehKg']*self.gravityMPerSec2))*self.gravityMPerSec2
        self.maxRegenKwh = 0.5*self.veh['vehKg']*(27**2)/(3600*1000)

        #############################
        ### Initialize Variables  ###
        #############################

        ### Drive Cycle [the following six varaibles are fixed array, which features the predefined road
		#   and cycle conditions. They never change during step iteration. We will us the self.steps as
		#   the index to get the value from the array every time the step function is called]
        self.cycSecs = np.copy( self.cyc['cycSecs'] )
        self.cycMps = np.copy( self.cyc['cycMps'] )
        self.cycGrade = np.copy( self.cyc['cycGrade'] )
        self.cycRoadType = np.copy( self.cyc['cycRoadType'] )
        self.cycMph = [x * self.mphPerMps for x in self.cyc['cycMps']]
        self.secs = np.insert(np.diff(self.cycSecs),0,0)

		# steps indicate the number of while iterations, which is the same meaning of "i" in original for loop
		self.steps=1

        ### Component Limits
        self.curMaxFsKwOut = 0
        self.fcTransLimKw = 0
        self.fcFsLimKw = 0
        self.fcMaxKwIn = 0
        self.curMaxFcKwOut = 0
        self.essCapLimDischgKw = 0
        self.curMaxEssKwOut = 0
        self.curMaxAvailElecKw = 0
        self.essCapLimChgKw = 0
        self.curMaxEssChgKw = 0
        self.curMaxRoadwayChgKw = np.interp( self.cycRoadType, self.veh['MaxRoadwayChgKw_Roadway'], self.veh['MaxRoadwayChgKw'] )
        self.curMaxElecKw = 0
        self.mcElecInLimKw = 0
        self.mcTransiLimKw = 0
        self.curMaxMcKwOut = 0
        self.essLimMcRegenPercKw = 0
        self.essLimMcRegenKw = 0
        self.curMaxMechMcKwIn = 0
        self.curMaxTransKwOut = 0

        ### Drive Train
        self.cycDragKw = 0
        self.cycAccelKw = 0
        self.cycAscentKw = 0
        self.cycTracKwReq = 0
        self.curMaxTracKw = 0
        self.spareTracKw = 0
        self.cycRrKw = 0
        self.cycWheelRadPerSec = 0
        self.cycTireInertiaKw = 0
        self.cycWheelKwReq = 0
        self.regenContrLimKwPerc = 0
        self.cycRegenBrakeKw = 0
        self.cycFricBrakeKw = 0
        self.cycTransKwOutReq = 0
        self.cycMet = 0
        self.transKwOutAch = 0
        self.curSocTarget = 0
        self.minMcKw2HelpFc = 0
        self.mcMechKwOutAch = 0
        self.mcElecKwInAch = 0
        self.auxInKw = 0

        #roadwayMaxEssChg = [0]*len(cycSecs)
        self.roadwayChgKwOutAch = 0
        self.minEssKw2HelpFc = 0
        self.essKwOutAch = 0
        self.fcKwOutAch = 0
        self.fcKwOutAch_pct = 0
        self.fcKwInAch = 0
        self.fsKwOutAch = 0
        self.fsKwhOutAch = 0
        self.essCurKwh = 0

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = 0
        self.essRegenBufferDischgKw = 0
        self.maxEssRegenBufferChgKw = 0
        self.essAccelBufferChgKw = 0
        self.accelBufferSoc = 0
        self.maxEssAccelBufferDischgKw = 0
        self.essAccelRegenDischgKw = 0
        self.mcElectInKwForMaxFcEff = 0
        self.electKwReq4AE = 0
        self.canPowerAllElectrically = 0
        self.desiredEssKwOutForAE = 0
        self.essAEKwOut = 0
        self.erAEKwOut = 0
        self.essDesiredKw4FcEff = 0
        self.essKwIfFcIsReq = 0
        self.curMaxMcElecKwIn = 0
        self.fcKwGapFrEff = 0
        self.erKwIfFcIsReq = 0
        self.mcElecKwInIfFcIsReq = 0
        self.mcKwIfFcIsReq = 0
        self.fcForcedOn = np.full(1,False)
        self.fcForcedState = 0
        self.mcMechKw4ForcedFc = 0
        self.fcTimeOn = 0
        self.prevfcTimeOn = 0

        ### Additional Variables
        self.mphAch = 0
        self.distMeters = 0
        self.distMiles = 0
        self.highAccFcOnTag = 0
        self.reachedBuff = 0
        self.maxTracMps = 0
        self.addKwh = 0
        self.dodCycs = 0
        self.essPercDeadArray = 0
        self.dragKw = 0
        self.essLossKw = 0
        self.accelKw = 0
        self.ascentKw = 0
        self.rrKw = 0
        self.motor_index_debug = 0
        self.debug_flag = 0

        ############################
        ###  Assign First Value  ###
        ############################

        ### Drive Train
        self.cycMet = 1
        self.curSocTarget = self.veh['maxSoc']
        self.essCurKwh = self.initSoc*self.veh['maxEssKwh']
        self.soc = self.initSoc

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, s_num):
        # Reset the state of the environment to an initial state
        if s_num == 2:
            transKwInAch = 0
            mpsAch = 0
            self.state = np.array([transKwInAch, mpsAch])
        if s_num == 3:
            transKwInAch = 0
            mpsAch = 0
            soc = 0
            self.state = np.array([transKwInAch, mpsAch, soc])

        return self.state


    def step(self, action):
		### previous step value cache
		### Component Limits
        curMaxFsKwOut = self.curMaxFsKwOut
        fcTransLimKw = self.fcTransLimKw
        fcFsLimKw = self.fcFsLimKw
        fcMaxKwIn = self.fcMaxKwIn
        curMaxFcKwOut = self.curMaxFcKwOut
        essCapLimDischgKw  = self.essCapLimDischgKw
        curMaxEssKwOut = self.curMaxEssKwOut
        curMaxAvailElecKw  = self.curMaxAvailElecKw
        essCapLimChgKw = self.essCapLimChgKw
        curMaxEssChgKw  = self.curMaxEssChgKw
        curMaxElecKw  = self.curMaxElecKw
        mcElecInLimKw = self.mcElecInLimKw
        mcTransiLimKw  = self.mcTransiLimKw
        curMaxMcKwOut = self.curMaxMcKwOut
        essLimMcRegenPercKw  = self.essLimMcRegenPercKw
        essLimMcRegenKw = self.essLimMcRegenKw
        curMaxMechMcKwIn = self.curMaxMechMcKwIn
        curMaxTransKwOut = self.curMaxTransKwOut

        ### Drive Train
        cycDragKw = self.cycDragKw
        cycAccelKw = self.cycAccelKw
        cycAscentKw = self.cycAscentKw
        cycTracKwReq = self.cycTracKwReq
        curMaxTracKw = self.curMaxTracKw
        spareTracKw = self.spareTracKw
        cycRrKw  = self.cycRrKw
        cycWheelRadPerSec  = self.cycWheelRadPerSec
        cycTireInertiaKw  = self.cycTireInertiaKw
        cycWheelKwReq = self.cycWheelKwReq
        regenContrLimKwPerc = self.regenContrLimKwPerc
        cycRegenBrakeKw = self.cycRegenBrakeKw
        cycFricBrakeKw = self.cycFricBrakeKw
        cycTransKwOutReq = self.cycTransKwOutReq
        cycMet = self.cycMet
        transKwOutAch = self.transKwOutAch
        transKwInAch = self.transKwInAch
        curSocTarget = self.curSocTarget
        minMcKw2HelpFc = self.minMcKw2HelpFc
        mcMechKwOutAch = self.mcMechKwOutAch
        mcElecKwInAch = self.mcElecKwInAch
        auxInKw = self.auxInKw

        #roadwayMaxEssChg = [0]*len(cycSecs)
        roadwayChgKwOutAch = self.roadwayChgKwOutAch
        minEssKw2HelpFc = self.minEssKw2HelpFc
        essKwOutAch = self.essKwOutAch
        fcKwOutAch = self.fcKwOutAch
        fcKwOutAch_pct = self.fcKwOutAch_pct
        fcKwInAch = self.fcKwInAch
        fsKwOutAch = self.fsKwOutAch
        fsKwhOutAch = self.fsKwhOutAch
        essCurKwh = self.essCurKwh

        # Vehicle Attributes, Control Variables
        regenBufferSoc = self.regenBufferSoc
        essRegenBufferDischgKw = self.essRegenBufferDischgKw
        maxEssRegenBufferChgKw = self.maxEssRegenBufferChgKw
        essAccelBufferChgKw = self.essAccelBufferChgKw
        accelBufferSoc = self.accelBufferSoc
        maxEssAccelBufferDischgKw = self.maxEssAccelBufferDischgKw
        essAccelRegenDischgKw = self.essAccelRegenDischgKw
        mcElectInKwForMaxFcEff = self.mcElectInKwForMaxFcEff
        electKwReq4AE = self.electKwReq4AE
        canPowerAllElectrically = self.canPowerAllElectrically
        desiredEssKwOutForAE = self.desiredEssKwOutForAE
        essAEKwOut = self.essAEKwOut
        erAEKwOut = self.erAEKwOut
        essDesiredKw4FcEff = self.essDesiredKw4FcEff
        essKwIfFcIsReq = self.essKwIfFcIsReq
        curMaxMcElecKwIn = self.curMaxMcElecKwIn
        fcKwGapFrEff = self.fcKwGapFrEff
        erKwIfFcIsReq = self.erKwIfFcIsReq
        mcElecKwInIfFcIsReq = self.mcElecKwInIfFcIsReq
        mcKwIfFcIsReq = self.mcKwIfFcIsReq
        fcForcedOn = self.fcForcedOn
        fcForcedState = self.fcForcedState
        mcMechKw4ForcedFc = self.mcMechKw4ForcedFc
        fcTimeOn = self.fcTimeOn
        prevfcTimeOn = self.prevfcTimeOn

        ### Additional Variables
        mphAch = self.mphAch
        mpsAch = self.mpsAch
        distMeters = self.distMeters
        distMiles = self.distMiles
        highAccFcOnTag = self.highAccFcOnTag
        reachedBuff = self.reachedBuff
        maxTracMps = self.maxTracMps
        addKwh = self.addKwh
        dodCycs = self.dodCycs
        essPercDeadArray = self.essPercDeadArray
        dragKw = self.dragKw
        essLossKw = self.essLossKw
        accelKw = self.accelKw
        ascentKw = self.ascentKw
        rrKw = self.rrKw
        motor_index_debug = self.motor_index_debug
        debug_flag = self.debug_flag

		### Drive Train
        essCurKwh = self.essCurKwh
        soc = self.soc

        ### Misc calcs
        if self.veh['noElecAux']=='TRUE':
            self.auxInKw = self.veh['auxKw']/self.veh['altEff']
        else:
            self.auxInKw = self.veh['auxKw']

        if self.soc<(self.veh['minSoc']+self.veh['percHighAccBuf']):
            self.reachedBuff = 0
        else:
            self.reachedBuff = 1
        if self.soc<self.veh['minSoc'] or (self.highAccFcOnTag==1 and self.reachedBuff==0):
            self.highAccFcOnTag = 1
        else:
            self.highAccFcOnTag = 0
        self.maxTracMps = self.mpsAch+(self.maxTracMps2*self.secs[self.steps])

        ### Component Limits
        self.curMaxFsKwOut = min( self.veh['maxFuelStorKw'] , self.fsKwOutAch + ((self.veh['maxFuelStorKw']/self.veh['fuelStorSecsToPeakPwr'])*(self.secs[self.steps])))
        self.fcTransLimKw = self.fcKwOutAch + ((self.veh['maxFuelConvKw']/self.veh['fuelConvSecsToPeakPwr'])*(self.secs[self.steps]))

        self.fcMaxKwIn = min(self.curMaxFsKwOut, veh['maxFuelStorKw'])
        self.fcFsLimKw = self.veh['fcMaxOutkW']
        self.curMaxFcKwOut = min(veh['maxFuelConvKw'],self.fcFsLimKw,self.fcTransLimKw)

        if veh['maxEssKwh']==0 or soc<veh['minSoc']:
            self.essCapLimDischgKw = 0.0

        else:
            self.essCapLimDischgKw = (veh['maxEssKwh']*np.sqrt(veh['essRoundTripEff']))*3600.0*(soc-veh['minSoc'])/(self.secs[self.steps]))
        curMaxEssKwOut[i] = min(veh['maxEssKw'],self.essCapLimDischgKw)

        if  veh['maxEssKwh'] == 0 or veh['maxEssKw'] == 0:
            self.essCapLimChgKw = 0

        else:
            self.essCapLimChgKw = max(((veh['maxSoc']-soc)*veh['maxEssKwh']*(1/np.sqrt(veh['essRoundTripEff'])))/((self.secs[self.steps]))*(1/3600.0)),0)

        self.curMaxEssChgKw = min(self.essCapLimChgKw,veh['maxEssKw'])

        if veh['fcEffType']==4:
            self.curMaxElecKw = self.curMaxFcKwOut+self.curMaxRoadwayChgKw+self.curMaxEssKwOut-self.auxInKw

        else:
            self.curMaxElecKw = self.curMaxRoadwayChgKw+self.curMaxEssKwOut-self.auxInKw

        self.curMaxAvailElecKw = min(self.curMaxElecKw, veh['mcMaxElecInKw'])

        if self.curMaxElecKw>0:
            if self.curMaxAvailElecKw == max(veh['mcKwInArray']):
                self.mcElecInLimKw = min(veh['mcKwOutArray'][len(veh['mcKwOutArray'])-1],veh['maxMotorKw'])
            else:
                self.mcElecInLimKw = min(veh['mcKwOutArray'][np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,self.curMaxAvailElecKw))-1],veh['maxMotorKw'])
        else:
            self.mcElecInLimKw = 0.0

        self.mcTransiLimKw = abs(mcMechKwOutAch)+((veh['maxMotorKw']/veh['motorSecsToPeakPwr'])*(self.secs[self.steps])))
        self.curMaxMcKwOut = max(min(self.mcElecInLimKw,self.mcTransiLimKw,veh['maxMotorKw']),-veh['maxMotorKw'])

        if self.curMaxMcKwOut == 0:
            self.curMaxMcElecKwIn = 0
        else:
            if self.curMaxMcKwOut == veh['maxMotorKw']:
                self.curMaxMcElecKwIn = self.curMaxMcKwOut/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                self.curMaxMcElecKwIn = self.curMaxMcKwOut/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.curMaxMcKwOut))-1)]

        if veh['maxMotorKw']==0:
            self.essLimMcRegenPercKw = 0.0

        else:
            self.essLimMcRegenPercKw = min((self.curMaxEssChgKw+self.auxInKw)/veh['maxMotorKw'],1)
        if self.curMaxEssChgKw==0:
            self.essLimMcRegenKw = 0.0

        else:
            if veh['maxMotorKw'] == self.curMaxEssChgKw-self.curMaxRoadwayChgKw:
                self.essLimMcRegenKw = min(veh['maxMotorKw'],self.curMaxEssChgKw/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1])
            else:
                self.essLimMcRegenKw = min(veh['maxMotorKw'],self.curMaxEssChgKw/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.curMaxEssChgKw-self.curMaxRoadwayChgKw))-1)])

        self.curMaxMechMcKwIn = min(self.essLimMcRegenKw,veh['maxMotorKw'])
        self.curMaxTracKw = (((veh['wheelCoefOfFric']*veh['driveAxleWeightFrac']*veh['vehKg']*gravityMPerSec2)/(1+((veh['vehCgM']*veh['wheelCoefOfFric'])/veh['wheelBaseM'])))/1000.0)*(self.maxTracMps)

        if veh['fcEffType']==4:

            if veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or self.highAccFcOnTag==1:
                self.curMaxTransKwOut = min((self.curMaxMcKwOut-self.auxInKw)*veh['transEff'],self.curMaxTracKw/veh['transEff'])
                self.debug_flag = 1

            else:
                self.curMaxTransKwOut = min((self.curMaxMcKwOut-min(self.curMaxElecKw,0))*veh['transEff'],self.curMaxTracKw/veh['transEff'])
                self.debug_flag = 2

        else:

            if veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or self.highAccFcOnTag==1:
                self.curMaxTransKwOut = min((self.curMaxMcKwOut+self.curMaxFcKwOut-auxInKw)*veh['transEff'],self.curMaxTracKw/veh['transEff'])
                self.debug_flag = 3

            else:
                self.curMaxTransKwOut = min((self.curMaxMcKwOut+self.curMaxFcKwOut-min(self.curMaxElecKw,0))*veh['transEff'],self.curMaxTracKw/veh['transEff'])
                self.debug_flag = 4

        ### Cycle Power
        self.cycDragKw = 0.5*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*(((mpsAch+self.cycMps[self.steps])/2.0)**3)/1000.0
        self.cycAccelKw = (veh['vehKg']/(2.0*(self.secs[self.steps]))))*((self.cycMps[self.steps]**2)-(mpsAch**2))/1000.0
        self.cycAscentKw = gravityMPerSec2*np.sin(np.arctan(self.cycGrade))*veh['vehKg']*((mpsAch+self.cycMps[self.steps])/2.0)/1000.0
        self.cycTracKwReq = self.cycDragKw + self.cycAccelKw + self.cycAscentKw
        self.spareTracKw = self.curMaxTracKw-self.cycTracKwReq
        self.cycRrKw = gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*((mpsAch+self.cycMps[self.steps])/2.0)/1000.0
        self.cycWheelRadPerSec = self.cycMps[self.steps]/veh['wheelRadiusM']
        self.cycTireInertiaKw = (((0.5)*veh['wheelInertiaKgM2']*(veh['numWheels']*(self.cycWheelRadPerSec**2.0))/self.secs[self.steps]))-((0.5)*veh['wheelInertiaKgM2']*(veh['numWheels']*((mpsAch/veh['wheelRadiusM'])**2.0))/self.secs[self.steps]))/1000.0

        self.cycWheelKwReq = self.cycTracKwReq + self.cycRrKw + self.cycTireInertiaKw
        self.regenContrLimKwPerc = veh['maxRegen']/(1+veh['regenA']*np.exp(-veh['regenB']*((self.cycMph+mpsAch*mphPerMps)/2.0+1-0)))
        self.cycRegenBrakeKw = max(min(self.curMaxMechMcKwIn*veh['transEff'],self.regenContrLimKwPerc*-self.cycWheelKwReq),0)
        self.cycFricBrakeKw = -min(self.cycRegenBrakeKw+self.cycWheelKwReq,0)
        self.cycTransKwOutReq = self.cycWheelKwReq + self.cycFricBrakeKw

        if self.cycTransKwOutReq<=self.curMaxTransKwOut:
            self.cycMet = 1
            self.transKwOutAch = self.cycTransKwOutReq
        else:
            self.cycMet = -1
            self.transKwOutAch = self.curMaxTransKwOut

        ################################
        ###   Speed/Distance Calcs   ###
        ################################

        #Cycle is met
        if self.cycMet==1:
            self.mpsAch = self.cycMps[self.steps]

        #Cycle is not met
        else:
            Drag3 = (1.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']
            Accel2 = veh['vehKg']/(2.0*(self.secs[self.steps])))
            Drag2 = (3.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*mpsAch
            Wheel2 = 0.5*veh['wheelInertiaKgM2']*veh['numWheels']/(self.secs[self.steps])*(veh['wheelRadiusM']**2))
            Drag1 = (3.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*((mpsAch)**2)
            Roll1 = (gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']/2.0)
            Ascent1 = (gravityMPerSec2*np.sin(np.arctan(self.cycGrade[self.steps]))*veh['vehKg']/2.0)
            Accel0 = -(veh['vehKg']*((mpsAch)**2))/(2.0*(self.secs[self.steps])))
            Drag0 = (1.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*((mpsAch)**3)
            Roll0 = (gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*mpsAch/2.0)
            Ascent0 = (gravityMPerSec2*np.sin(np.arctan(self.cycGrade[self.steps]))*veh['vehKg']*mpsAch/2.0)
            Wheel0 = -((0.5*veh['wheelInertiaKgM2']*veh['numWheels']*(mpsAch**2))/(self.secs[self.steps])*(veh['wheelRadiusM']**2)))

            Total3 = Drag3/1e3
            Total2 = (Accel2+Drag2+Wheel2)/1e3
            Total1 = (Drag1+Roll1+Ascent1)/1e3
            Total0 = (Accel0+Drag0+Roll0+Ascent0+Wheel0)/1e3 - self.stepscurMaxTransKwOut

            Total = [Total3,Total2,Total1,Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin( abs(self.cycMps[self.steps] - Total_roots) )
            self.mpsAch = Total_roots[ind]

        self.mphAch = self.mpsAch*mphPerMps
        self.distMeters = self.mpsAch*self.secs[self.steps])
        self.distMiles = self.distMeters*(1.0/metersPerMile)

        ### Drive Train
        if self.transKwOutAch>0:
            self.transKwInAch = self.transKwOutAch/veh['transEff']
        else:
            self.transKwInAch = self.transKwOutAch*veh['transEff']

        if self.cycMet==1:

            if veh['fcEffType']==4:
                self.minMcKw2HelpFc = max(self.transKwInAch, -self.curMaxMechMcKwIn)

            else:
                self.minMcKw2HelpFc = max(self.transKwInAch - self.curMaxFcKwOut, -self.curMaxMechMcKwIn)
        else:
            self.minMcKw2HelpFc = max(self.curMaxMcKwOut, -self.curMaxMechMcKwIn)

        if veh['noElecSys'] == 'TRUE':
            self.regenBufferSoc = 0

        elif veh['chargingOn']:
            self.regenBufferSoc = max(veh['maxSoc'] - (maxRegenKwh/veh['maxEssKwh']), (veh['maxSoc']+veh['minSoc'])/2)

        else:
            self.regenBufferSoc = max(((veh['maxEssKwh']*veh['maxSoc'])-(0.5*veh['vehKg']*(self.cycMps[self.steps]**2)*(1.0/1000)*(1.0/3600)*veh['motorPeakEff']*veh['maxRegen']))/veh['maxEssKwh'],veh['minSoc'])


        self.essRegenBufferDischgKw = min(self.curMaxEssKwOut, max(0,(soc-self.regenBufferSoc)*veh['maxEssKwh']*3600/self.secs[self.steps])))
        self.maxEssRegenBufferChgKw = min(max(0,(self.regenBufferSoc-soc)*veh['maxEssKwh']*3600.0/self.secs[self.steps])),self.curMaxEssChgKw)

        if veh['noElecSys']=='TRUE':
            self.accelBufferSoc = 0

        else:
            self.accelBufferSoc = min(max((((((((veh['maxAccelBufferMph']*(1/mphPerMps))**2))-((self.cycMps[self.steps]**2)))/(((veh['maxAccelBufferMph']*(1/mphPerMps))**2)))*(min(veh['maxAccelBufferPercOfUseableSoc']*(veh['maxSoc']-veh['minSoc']),maxRegenKwh/veh['maxEssKwh'])*veh['maxEssKwh']))/veh['maxEssKwh'])+veh['minSoc'],veh['minSoc']), veh['maxSoc'])

        self.essAccelBufferChgKw = max(0,(self.accelBufferSoc - soc)*veh['maxEssKwh']*3600.0/self.secs[self.steps]))
        self.maxEssAccelBufferDischgKw = min(max(0, (soc-self.accelBufferSoc)*veh['maxEssKwh']*3600/self.secs[self.steps])),self.curMaxEssKwOut)

        if self.regenBufferSoc < self.accelBufferSoc:
            self.essAccelRegenDischgKw = max(min(((soc-(self.regenBufferSoc+self.accelBufferSoc)/2)*veh['maxEssKwh']*3600.0)/self.secs[self.steps]),self.curMaxEssKwOut),-self.curMaxEssChgKw)

        elif soc>self.regenBufferSoc:
            self.essAccelRegenDischgKw = max(min(self.essRegenBufferDischgKw,self.curMaxEssKwOut),-self.curMaxEssChgKw)

        elif soc<self.accelBufferSoc:
            self.essAccelRegenDischgKw = max(min(-1.0*self.essAccelBufferChgKw,self.curMaxEssKwOut),-self.curMaxEssChgKw)

        else:
            self.essAccelRegenDischgKw = max(min(0,self.curMaxEssKwOut),-self.curMaxEssChgKw)

        self.fcKwGapFrEff = abs(self.transKwOutAch-veh['maxFcEffKw'])

        if veh['noElecSys']=='TRUE':
            self.mcElectInKwForMaxFcEff = 0

        elif self.transKwOutAch<veh['maxFcEffKw']:

            if self.fcKwGapFrEff == veh['maxMotorKw']:
                self.mcElectInKwForMaxFcEff = self.fcKwGapFrEff/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]*-1
            else:
                self.mcElectInKwForMaxFcEff = self.fcKwGapFrEff/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.fcKwGapFrEff))-1)]*-1

        else:

            if self.fcKwGapFrEff == veh['maxMotorKw']:
                self.mcElectInKwForMaxFcEff = veh['mcKwInArray'][len(veh['mcKwInArray'])-1]
            else:
                self.mcElectInKwForMaxFcEff = veh['mcKwInArray'][np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.fcKwGapFrEff))-1]

        if veh['noElecSys']=='TRUE':
            self.electKwReq4AE = 0

        elif self.transKwInAch > 0:
            if self.transKwInAch == veh['maxMotorKw']:

                self.electKwReq4AE = self.transKwInAch/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]+self.auxInKw
            else:
                self.electKwReq4AE = self.transKwInAch/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.transKwInAch))-1)]+self.auxInKw

        else:
            self.electKwReq4AE = 0

        self.prevfcTimeOn = fcTimeOn

        if veh['maxFuelConvKw']==0:
            self.canPowerAllElectrically = self.accelBufferSoc<soc and self.transKwInAch<=self.curMaxMcKwOut and (self.electKwReq4AE<self.curMaxElecKw or veh['maxFuelConvKw']==0)

        else:
            self.canPowerAllElectrically = self.accelBufferSoc<soc and self.transKwInAch<=self.curMaxMcKwOut and (self.electKwReq4AE<self.curMaxElecKw or veh['maxFuelConvKw']==0) and (self.cycMph[self.steps]-0.00001<=veh['mphFcOn'] or veh['chargingOn']) and self.electKwReq4AE<=veh['kwDemandFcOn']

        if self.canPowerAllElectrically:

            if self.transKwInAch<+self.auxInKw:
                self.desiredEssKwOutForAE = self.auxInKw+self.transKwInAch

            elif self.regenBufferSoc<self.accelBufferSoc:
                self.desiredEssKwOutForAE = self.essAccelRegenDischgKw

            elif soc>self.regenBufferSoc:
                self.desiredEssKwOutForAE = self.essRegenBufferDischgKw
            elif soc<self.accelBufferSoc:
                self.desiredEssKwOutForAE = -self.essAccelBufferChgKw

            else:
                self.desiredEssKwOutForAE = self.transKwInAch+self.auxInKw-self.curMaxRoadwayChgKw

        else:
            self.desiredEssKwOutForAE = 0

        if self.canPowerAllElectrically:
            self.essAEKwOut = max(-self.curMaxEssChgKw,-self.maxEssRegenBufferChgKw,min(0,self.curMaxRoadwayChgKw-(self.transKwInAch+self.auxInKw)),min(self.curMaxEssKwOut,self.desiredEssKwOutForAE))

        else:
            self.essAEKwOut = 0

        self.erAEKwOut = min(max(0,self.transKwInAch+self.auxInKw-self.essAEKwOut),self.curMaxRoadwayChgKw)

        if self.prevfcTimeOn>0 and self.prevfcTimeOn<veh['minFcTimeOn']-self.secs[self.steps]):
            self.fcForcedOn = True
        else:
            self.fcForcedOn = False

#####################################  ECMS stragegy was here  ##############################################


        if self.fcForcedOn==False or self.canPowerAllElectrically==False:
            self.fcForcedState = 1
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = 0

        elif self.transKwInAch<0:
            self.fcForcedState = 2
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = transKwInAch[i]

        elif veh['maxFcEffKw']==self.transKwInAch:
            self.fcForcedState = 3
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = 0

        elif veh['idleFcKw'] > self.transKwInAch and self.cycAccelKw >=0:
            self.fcForcedState = 4
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = transKwInAch[i] - veh['idleFcKw']

        elif veh['maxFcEffKw']>self.transKwInAch:
            self.fcForcedState = 5
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = 0

        else:
            ##################################################################################################
            ## max ICE efficiency, the rest is filled in by EM
            self.fcForcedState = 6
#             mcMechKw4ForcedFc[i] = (soc[i-1]-soc_ref)*soc_slop
#             mcMechKw4ForcedFc[i] = transKwInAch[i] - veh['maxFcEffKw']

        if (-self.mcElectInKwForMaxFcEff-self.curMaxRoadwayChgKw)>0:
            self.essDesiredKw4FcEff = (-self.mcElectInKwForMaxFcEff-self.curMaxRoadwayChgKw) * veh['essDischgToFcMaxEffPerc']

        else:
            self.essDesiredKw4FcEff = (-self.mcElectInKwForMaxFcEff-self.curMaxRoadwayChgKw) * veh['essChgToFcMaxEffPerc']

        if self.accelBufferSoc>self.regenBufferSoc:
            self.essKwIfFcIsReq = min(self.curMaxEssKwOut,veh['mcMaxElecInKw']+self.auxInKw,self.curMaxMcElecKwIn+self.auxInKw, max(-self.curMaxEssChgKw, self.essAccelRegenDischgKw))

        elif self.essRegenBufferDischgKw>0:
            self.essKwIfFcIsReq = min(self.curMaxEssKwOut,veh['mcMaxElecInKw']+self.auxInKw,self.curMaxMcElecKwIn+self.auxInKw, max(-self.curMaxEssChgKw, min(self.essAccelRegenDischgKw,self.mcElecInLimKw+self.auxInKw, max(self.essRegenBufferDischgKw,self.essDesiredKw4FcEff))))

        elif self.essAccelBufferChgKw>0:
            self.essKwIfFcIsReq = min(self.curMaxEssKwOut,veh['mcMaxElecInKw']+self.auxInKw,self.curMaxMcElecKwIn+self.auxInKw, max(-self.curMaxEssChgKw, max(-1*self.maxEssRegenBufferChgKw, min(-self.essAccelBufferChgKw,self.essDesiredKw4FcEff))))


        elif self.essDesiredKw4FcEff>0:
            self.essKwIfFcIsReq = min(self.curMaxEssKwOut,veh['mcMaxElecInKw']+self.auxInKw,self.curMaxMcElecKwIn+self.auxInKw, max(-self.curMaxEssChgKw, min(self.essDesiredKw4FcEff,self.maxEssAccelBufferDischgKw)))

        else:
            self.essKwIfFcIsReq = min(self.curMaxEssKwOut,veh['mcMaxElecInKw']+self.auxInKw,self.curMaxMcElecKwIn+self.auxInKw, max(-self.curMaxEssChgKw, max(self.essDesiredKw4FcEff,-self.maxEssRegenBufferChgKw)))

        self.erKwIfFcIsReq = max(0,min(self.curMaxRoadwayChgKw,self.curMaxMechMcKwIn,self.essKwIfFcIsReq-self.mcElecInLimKw+self.auxInKw))

        self.mcElecKwInIfFcIsReq = self.essKwIfFcIsReq+self.erKwIfFcIsReq-self.auxInKw

        if veh['noElecSys']=='TRUE':
            self.mcKwIfFcIsReq = 0

        elif  self.mcElecKwInIfFcIsReq == 0:
            self.mcKwIfFcIsReq = 0

        elif self.mcElecKwInIfFcIsReq > 0:

            if self.mcElecKwInIfFcIsReq == max(veh['mcKwInArray']):
                 self.mcKwIfFcIsReq = self.mcElecKwInIfFcIsReq*veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                 self.mcKwIfFcIsReq = self.mcElecKwInIfFcIsReq*veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,self.mcElecKwInIfFcIsReq))-1)]

        else:
            if self.mcElecKwInIfFcIsReq*-1 == max(veh['mcKwInArray']):
                self.mcKwIfFcIsReq = self.mcElecKwInIfFcIsReq/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                self.mcKwIfFcIsReq = self.mcElecKwInIfFcIsReq/(veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,self.mcElecKwInIfFcIsReq*-1))-1)])

        if veh['maxMotorKw']==0:
            self.mcMechKwOutAch = 0

        elif self.fcForcedOn==True and self.canPowerAllElectrically==True and (veh['vehPtType']==2.0 or veh['vehPtType']==3.0) and veh['fcEffType']!=4:
            self.mcMechKwOutAch =  self.mcMechKw4ForcedFc

        elif self.transKwInAch<=0:
            if veh['fcEffType']!=4 and veh['maxFuelConvKw']> 0:
                if self.canPowerAllElectrically == 1:
                    self.mcMechKwOutAch = -min(self.curMaxMechMcKwIn,-self.transKwInAch)
                else:
                    self.mcMechKwOutAch = min(-min(self.curMaxMechMcKwIn, -self.transKwInAch),max(-self.curMaxFcKwOut, self.mcKwIfFcIsReq))
            else:
                self.mcMechKwOutAch = min(-min(self.curMaxMechMcKwIn,-self.transKwInAch),-self.transKwInAch)

        elif self.canPowerAllElectrically == 1:
            self.mcMechKwOutAch = self.transKwInAch

        else:
            ################################################################################################
            ################################################################################################
            ################################################################################################
            #
            self.mcMechKwOutAch = self.mcMechKw4ForcedFc
#             mcMechKwOutAch[i] = max(minMcKw2HelpFc[i],mcKwIfFcIsReq[i])

        if self.mcMechKwOutAch==0:
            self.mcElecKwInAch = 0.0
            self.motor_index_debug = 0

        elif self.mcMechKwOutAch<0:

            if self.mcMechKwOutAch*-1 == max(veh['mcKwInArray']):
                self.mcElecKwInAch = self.mcMechKwOutAch*veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                self.mcElecKwInAch = self.mcMechKwOutAch*veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,self.mcMechKwOutAch*-1))-1)]

        else:
            if veh['maxMotorKw'] == self.mcMechKwOutAch:
                self.mcElecKwInAch = self.mcMechKwOutAch/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                self.mcElecKwInAch = self.mcMechKwOutAch/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,self.mcMechKwOutAch))-1)]

        if self.curMaxRoadwayChgKw == 0:
            self.roadwayChgKwOutAch = 0

        elif veh['fcEffType']==4:
            self.roadwayChgKwOutAch = max(0, self.mcElecKwInAch, self.maxEssRegenBufferChgKw, self.essRegenBufferDischgKw, self.curMaxRoadwayChgKw)

        elif self.canPowerAllElectrically == 1:
            self.roadwayChgKwOutAch = self.erAEKwOut

        else:
            self.roadwayChgKwOutAch = self.erKwIfFcIsReq

        self.minEssKw2HelpFc = self.mcElecKwInAch + self.auxInKw - self.curMaxFcKwOut - self.roadwayChgKwOutAch

        if veh['maxEssKw'] == 0 or veh['maxEssKwh'] == 0:
            self.essKwOutAch  = 0

        elif veh['fcEffType']==4:

            if self.transKwOutAch>=0:
                self.essKwOutAch = min(max(self.minEssKw2HelpFc,self.essDesiredKw4FcEff,self.essAccelRegenDischgKw),self.curMaxEssKwOut,self.mcElecKwInAch+self.auxInKw-self.roadwayChgKwOutAch)

            else:
                self.essKwOutAch = self.mcElecKwInAch+self.auxInKw-self.roadwayChgKwOutAch

        elif self.highAccFcOnTag==1 or veh['noElecAux']=='TRUE':
            self.essKwOutAch = self.mcElecKwInAch-self.roadwayChgKwOutAch

        else:
            self.essKwOutAch = self.mcElecKwInAch+self.auxInKw-self.roadwayChgKwOutAch

        if veh['maxFuelConvKw']==0:
            self.fcKwOutAch = 0

        elif veh['fcEffType']==4:
            self.fcKwOutAch = min(self.curMaxFcKwOut, max(0, self.mcElecKwInAch+self.auxInKw-self.essKwOutAch-self.roadwayChgKwOutAch))

        elif veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or self.highAccFcOnTag==1:
            self.fcKwOutAch = min(self.curMaxFcKwOut, max(0, self.transKwInAch-self.mcMechKwOutAch+self.auxInKw))

        else:
            self.fcKwOutAch = min(self.curMaxFcKwOut, max(0, self.transKwInAch-self.mcMechKwOutAch))

        if self.fcKwOutAch==0:
            self.fcKwInAch = 0.0
            self.fcKwOutAch_pct = 0

        if veh['maxFuelConvKw'] == 0:
            self.fcKwOutAch_pct = 0
        else:
            self.fcKwOutAch_pct = self.fcKwOutAch / veh['maxFuelConvKw']

        if self.fcKwOutAch == 0:
            self.fcKwInAch = 0
        else:
            if self.fcKwOutAch == veh['fcMaxOutkW']:
                self.fcKwInAch = self.fcKwOutAch/veh['fcEffArray'][len(veh['fcEffArray'])-1]
            else:
                self.fcKwInAch = self.fcKwOutAch/(veh['fcEffArray'][max(1,np.argmax(veh['fcKwOutArray']>min(self.fcKwOutAch,veh['fcMaxOutkW']-0.001))-1)])

        self.fsKwOutAch = np.copy( self.fcKwInAch )

        self.fsKwhOutAch = self.fsKwOutAch*self.secs[self.steps])*(1/3600.0)


        if veh['noElecSys']=='TRUE':
            self.essCurKwh = 0

        elif self.essKwOutAch<0:
            self.essCurKwh = essCurKwh-self.essKwOutAch*(self.secs[self.steps])/3600.0)*np.sqrt(veh['essRoundTripEff'])

        else:
            self.essCurKwh = essCurKwh-self.essKwOutAch*(self.secs[self.steps])/3600.0)*(1/np.sqrt(veh['essRoundTripEff']))

        if veh['maxEssKwh']==0:
            self.soc = 0.0

        else:
            self.soc = self.essCurKwh/veh['maxEssKwh']

        if self.canPowerAllElectrically==True and self.fcForcedOn==False and self.fcKwOutAch==0:
            self.fcTimeOn = 0
        else:
            self.fcTimeOn = fcTimeOn + self.secs[self.steps])

        ### Battery wear calcs

        if veh['noElecSys']!='TRUE':

            if self.essCurKwh>essCurKwh:
                self.addKwh = (self.essCurKwh-essCurKwh) + addKwh
            else:
                self.addKwh = 0

            if self.addKwh==0:
                self.dodCycs = addKwh/veh['maxEssKwh']
            else:
                self.dodCycs = 0

            if self.dodCycs!=0:
                self.essPercDeadArray = np.power(veh['essLifeCoefA'],1.0/veh['essLifeCoefB']) / np.power(self.dodCycs,1.0/veh['essLifeCoefB'])
            else:
                self.essPercDeadArray = 0

        ### Energy Audit Calculations
        self.dragKw = 0.5*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*(((mpsAch+self.mpsAch)/2.0)**3)/1000.0
        if veh['maxEssKw'] == 0 or veh['maxEssKwh']==0:
            self.essLossKw = 0
        elif self.essKwOutAch<0:
            self.essLossKw = -self.essKwOutAch - (-self.essKwOutAch*np.sqrt(veh['essRoundTripEff']))
        else:
            self.essLossKw = self.essKwOutAch*(1.0/np.sqrt(veh['essRoundTripEff']))-self.essKwOutAch
        self.accelKw = (veh['vehKg']/(2.0*(self.secs[self.steps]))))*((self.mpsAch**2)-(mpsAch**2))/1000.0
        self.ascentKw = gravityMPerSec2*np.sin(np.arctan(self.cycGrade[self.steps]))*veh['vehKg']*((mpsAch+self.mpsAch)/2.0)/1000.0
        self.rrKw = gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*((mpsAch+self.mpsAch)/2.0)/1000.0

        # reward calculation
        if self.mcMechKw4ForcedFc==0:
            gam=1
        else:
            gam = (np.sign(self.mcMechKw4ForcedFc)+1)*0.5
        eff_EM = np.interp(np.abs(self.mcMechKw4ForcedFc), mcKwOutArray, veh['mcFullEffArray'])
        m_equ_em_no_penalty =(s_EM*gam/eff_EM + s_EM*(1-gam)*eff_EM)*(self.mcMechKw4ForcedFc*1000)/Qhlv;
        x_soc = (soc-soc_l)/(soc_h-soc_l);
        weight_c=1;
        f_penalty=(1-(x_soc**3)/2)*weight_c; # penalty
        mc_m = f_penalty * m_equ_em_no_penalty


        # fuel consumption
        fsKwOutAch1 = self.transKwInAch-self.mcMechKw4ForcedFc
        fsKwhOutAch1 = fsKwOutAch1*self.secs[self.steps])*(1/3600.0)
        fs_m = fsKwhOutAch1*(1/kWhPerGGE)*kgPerGallon # [kg]
        tot_m = mc_m+fs_m
        self.reward = -w_m*tot_m/tot_m_norm + cost_bias - (1-w_m)*(0.6-self.soc)*(self.soc<0.6)/soc_error_norm;


		self.steps=self.steps+1

        return








    def render(self):


    def close(self):
