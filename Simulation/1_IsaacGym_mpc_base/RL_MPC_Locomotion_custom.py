import math
from MPC_Controller.Parameters import Parameters
from MPC_Controller.robot_runner.RobotRunnerFSM import RobotRunnerFSM
from MPC_Controller.robot_runner.RobotRunnerMin import RobotRunnerMin
from MPC_Controller.robot_runner.RobotRunnerPolicy import RobotRunnerPolicy
from MPC_Controller.common.Quadruped import RobotType
from MPC_Controller.utils import DTYPE, ControllerType
from RL_Environment import gamepad_reader # Original
# from RL_Environment import keyboard_reader
from isaacgym import gymapi
from RL_Environment.sim_utils import *
from argparse import ArgumentParser

from trajectory_lib import basic_commands

parser = ArgumentParser(prog="RL_MPC_LOCOMOTION")

parser.add_argument("--robot", default="Aliengo", choices=[name.title() for name in RobotType.__members__.keys()], help="robot types")
parser.add_argument("--mode", default="Fsm", choices=[name.title() for name in ControllerType.__members__.keys()], help="controller types")
parser.add_argument("--num-envs", type=int, default=1, help="the number of gym envs")
parser.add_argument("--render-fps", type=int, default=30, help="render fps")
parser.add_argument("--disable-gamepad", action="store_true")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--bridge", default=False)  # call basic_commands.py functions
parser.add_argument("--num-agents", type=int, default=1, help="the number of robots")

args = parser.parse_args()

use_gamepad = not args.disable_gamepad
debug_vis = False # draw ground normal vector

use_bridge = args.bridge

if use_gamepad:
    gamepad = gamepad_reader.Gamepad(vel_scale_x=2.5, vel_scale_y=1.5, vel_scale_rot=3.0)
    # gamepad = keyboard_reader.Gamepad(vel_scale_x=2.5, vel_scale_y=1.5, vel_scale_rot=3.0)

if use_bridge:
    bridge = basic_commands.movement_bridge()

def main():
    robot = RobotType[args.robot.upper()]
    dt =  Parameters.controller_dt
    gym = gymapi.acquire_gym()
    sim = acquire_sim(gym, dt)
    add_ground(gym, sim)
    # add_random_uniform_terrain(gym, sim)
    add_terrain(gym, sim, "slope")
    add_terrain(gym, sim, "stair", 3.95, True)
    # add_uneven_terrains(gym, sim)

    # set up the env grid
    num_envs = args.num_envs
    envs_per_row = int(math.sqrt(args.num_envs))
    env_spacing = 0.5
    # one actor per env 
    # envs, actors = create_envs(gym, sim, robot, num_envs, envs_per_row, env_spacing)          # Original
    num_agents = args.num_agents
    agents_spacing = 1
    # envs_per_row = int(math.sqrt(num_envs*num_agents))
    envs, actors = create_envs_multirobot(gym, sim, robot, num_agents, num_envs, envs_per_row, agents_spacing) # Custom

    # force_sensors = get_force_sensor(gym, envs, actors)
    cam_pos = gymapi.Vec3(2,2,2) # w.r.t target env
    viewer = add_viewer(gym, sim, envs[0], cam_pos)

    # Setup MPC Controller
    controllers = []
    for idx in range(num_envs):
        # configure the joints for effort control mode (once)
        for idx_agent in range(num_agents):
            props = gym.get_actor_dof_properties(envs[idx], actors[idx_agent])
            props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            props["stiffness"].fill(0.0)
            props["damping"].fill(0.0)
            gym.set_actor_dof_properties(envs[idx], actors[idx_agent], props)

            # Setup MPC Controller
            controller_type = ControllerType[args.mode.upper()]
            if controller_type is ControllerType.FSM:
                robotRunner = RobotRunnerFSM()
            elif controller_type is ControllerType.MIN:
                robotRunner = RobotRunnerMin()
            elif controller_type is ControllerType.POLICY:
                robotRunner = RobotRunnerPolicy(args.checkpoint)
            else:
                raise Exception("Invalid ControllerType!")

            robotRunner.init(robot)
            controllers.append(robotRunner)

    count = 0
    render_fps = args.render_fps
    render_count = int(1/render_fps/Parameters.controller_dt)

    # simulation loop
    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # current_time = gym.get_sim_time(sim)
        commands = np.zeros(3, dtype=DTYPE)
        if use_gamepad:
            lin_speed, ang_speed, e_stop = gamepad.get_command()
            Parameters.cmpc_gait = gamepad.get_gait()
            Parameters.control_mode = gamepad.get_mode()

            # print("Mode current = ", gamepad.get_mode())
            # print("STOP = ", e_stop)

            if not e_stop:
                commands = np.array([lin_speed[0], lin_speed[1], ang_speed], dtype=DTYPE)
                print("Command = ", commands)
        
        if use_bridge:
            # bridge.move_FB_cte(direction=True)
            # bridge.move_rotate_cte(direction=True)
            # Parameters.cmpc_gait = bridge.gait_list[0]
            # Parameters.control_mode = bridge.mode_list[1]

            bridge.move_list_sequence()
            Parameters.cmpc_gait = bridge.gait
            Parameters.control_mode = bridge.mode

            commands = np.array([bridge.vx, bridge.vy, bridge.wz], dtype=DTYPE)

        # run controllers
        # for idx, (env, actor, controller) in enumerate(zip(envs, actors, controllers)):
        for env in envs:
            for idx, (actor, controller) in enumerate(zip(actors, controllers)):
                dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
                body_idx = gym.find_actor_rigid_body_index(env, actor, controller._quadruped._bodyName, gymapi.DOMAIN_ACTOR)
                body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)[body_idx]
                legTorques = controller.run(dof_states, body_states, commands).astype(np.float32)
                gym.apply_actor_dof_efforts(env, actor, legTorques)

        if Parameters.locomotionUnsafe:
            gamepad.fake_event(ev_type='Key',code='BTN_TR',value=0)
            Parameters.locomotionUnsafe = False

        if debug_vis:
            pos_np = np.asarray([p for p in body_states["pose"]["p"]], dtype=np.float32)
            gym.add_lines(viewer, envs[0], 1, 
                [pos_np, pos_np + controllers[0]._stateEstimator.result.ground_normal_world], 
                [[255,0,0]])

        if count % render_count == 0:
            # update the viewer
            count = 0
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.clear_lines(viewer)

        # Wait for dt to elapse in real time.
        gym.sync_frame_time(sim)
        count += 1
        # print("Counter = ", count)

    if use_gamepad:
        gamepad.stop()
        # gamepad.read_thread.join()
        # print("Gamepad read thread killed!") # too slow

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__=="__main__":
    main()
