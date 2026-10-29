'''
This file is kind of like your typical 'const.py' in that it contains the parameters and settings for you
to edit to make variables AOI-dep or constant, for you to run with/out CBF, switching, lossless, etc.

When you are done editing here, then in terminal run 'main.py', which reads variables from this file directly.
'''


SAFE_SET_RADIUS=1.0
SAFE_SET_CENTER= 0 
SAFE_SET_EPSILON = 1e-2 

HUMAN_DESIRED_TRAJ_RADIUS = 1.5
SAFE_TOLERANCE = 1e-10
N_s = 4000
RNG_SEED = 888
AOI_THRESHOLD = 10 #threshold for when to activate safe mode in the safe mode switching case

### dont change these parameters ####
# the cbf modes are enumerated
USE_NO_CBF=0
USE_STANDARD_CBF=1
USE_AOI_CBF=2
USE_AOI_VEL_CBF=3

# and the binary modes are toggled using the bitwise XOR, ^1 
LOSSLESS = 0 # 1 := perfect ideal, 0:=delay+losing packets, etc
SWITCHING = 1 # 1:=when aoi threshold reaached switch to safe mode, 0:=never use the safe mode
#########################


### edit these for your simulation run: ###
SWITCHING_SIMULATION =[SWITCHING, SWITCHING^1]
CBF_SIMULATION = [USE_NO_CBF, USE_STANDARD_CBF, USE_AOI_CBF, USE_AOI_VEL_CBF]
LOSSLESS_SIMULATION=[LOSSLESS^1]
                       