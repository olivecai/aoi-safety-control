from experiment_go import *


from . import slack_velocity_with_aoi_gamma
from . import standard_with_aoi_gamma
from . import standard
from . import slack_position_with_aoi_gamma
from . import cbf_slew_u
from . import local_limit
from . import local_limit_with_velocity_slack

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
    if cbf_enum == USE_AOI_POS_CBF:
        make_cbf = slack_position_with_aoi_gamma.make
        get_u_cbf = slack_position_with_aoi_gamma.get_u_cbf
    if cbf_enum == USE_SLEW_U_CBF:
        make_cbf = cbf_slew_u.make
        get_u_cbf = cbf_slew_u.get_u_cbf
    if cbf_enum == USE_LOCAL_LIMIT:
        make_cbf=local_limit.make
        get_u_cbf=local_limit.get_u_cbf
    if cbf_enum == USE_LOCAL_LIMIT_WITH_VELOCITY_SLACK:
        make_cbf=local_limit_with_velocity_slack.make
        get_u_cbf=local_limit_with_velocity_slack.get_u_cbf
    return make_cbf,get_u_cbf