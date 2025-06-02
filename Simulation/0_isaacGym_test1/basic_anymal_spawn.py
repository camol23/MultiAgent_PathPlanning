import math
import random
from isaacgym import gymapi, gymutil

gym = gymapi.acquire_gym()

## get default set of parameters
sim_params = gymapi.SimParams()

## set common parameters
sim_params.dt = 1 / 60
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

## configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up axis
# plane_params.normal = gymapi.Vec3(0, 1, 0) # y-up axis
plane_params.distance = 0 # specify where the ground plane be placed

## create the ground plane
gym.add_ground(sim, plane_params)

num_envs = 3
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 4.0)

# asset_root = "./assets"
# asset_file = "mjcf/nv_ant.xml"
asset_root = "/home/camilo/repos/IsaacGym_repo/IsaacGymEnvs/assets"
# asset_file = "urdf/anymal_c/urdf/anymal_minimal.urdf"
# asset_file = "urdf/anymal_c/urdf/anymal.urdf"

asset_file = "urdf/anymal_c/urdf/anymal.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = True
# asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
asset_options.fix_base_link = False
asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.armature = 0.0
asset_options.thickness = 0.01
asset_options.disable_gravity = False

ant_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

## load asset with default control type of position for all joints
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = False
# ant_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

## create envs
for i_env in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(random.uniform(-spacing, spacing), random.uniform(-spacing/2, spacing/2), 2.0)
    initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    ant_actor = gym.create_actor(env, ant_asset, initial_pose, 'nv_ant')

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # step the rendering of physics results
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # sync_frame_time throttle down the simulation rate to real time
    gym.sync_frame_time(sim)