# 3D Simulation Code
---

### Results
---
#### Predefine Sequence
* IsaacGym simulation based on MPC-RL locomotion controller developed by Yulun Zhuang [^1] 


<img src="Images/short_sequence_init.gif" width=500>

<img src="Images/short_sequence_stop.gif" width=500>



### Folder structure
---
```
├── 0_isaacGym_test1                                    
│   ├── basic_anymal_spawn.py                       # First approach with no controller (minimal code) 
|   |
│   ├── Omni_IsaacGym_models_2022                   # DRL controller models taken from OmniverseNulceus data
│   │   ├── anymal.pth
│   │   ├── anymal_terrain.pth
│   │   ├── cartpole.pth
│   │   ├── nucleus_steps.txt
│   │   └── policy_1.pt
|   |
│   └── README.md
|
├── 1_IsaacGym_mpc_base                                             
│   ├── RL_MPC_Locomotion_custom.py                 # Multiple agents controller by a predefined sequence of movements
│   ├── sim_utils.py                                # create_envs_multirobot() function
|   |
│   └── trajectory_lib
│       ├── basic_commands.py                       # Wrapper for Trajectory planning algoritms
|
├── notes.txt                                       # Commands, extra steps, annotations and comments in the dev.
└── README.md

```



### Reference
---
[^1]: [MPC-RL locomotion controller developed by Yulun Zhuang](https://github.com/silvery107/rl-mpc-locomotion.git) [^1]


### Extra

* /IsaacGym_repo/rl-mpc-locomotion/RL_Environment/sim_utils.py
