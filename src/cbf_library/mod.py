from experiment_go import *


from . import slack_velocity_with_aoi_gamma
from . import standard_with_aoi_gamma
from . import standard

def instantiate_cbf(cbf_enum):
    if cbf_enum == USE_NO_CBF:
        make_cbf = None
        get_u_cbf = None
    if cbf_enum == USE_STANDARD_CBF:
        make_cbf = standard.make
        get_u_cbf = standard.get_u_cbf
    if cbf_enum == USE_AOI_CBF:
        make_cbf = standard_with_aoi_gamma.make
        get_u_cbf = standard_with_aoi_gamma.get_u_cbf
    if cbf_enum == USE_AOI_VEL_CBF:
         make_cbf = slack_velocity_with_aoi_gamma.make
         get_u_cbf = slack_velocity_with_aoi_gamma.get_u_cbf
    return make_cbf,get_u_cbf