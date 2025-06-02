# IsaacGym 

Testing with the IsaacGym basic templates.
    - The code has been edited to force the CPU execution.


### Results 
----
1) Main IsaacGym with controller taken from OmniIsaacGym. [^1]
```bash
 python train.py task=Anymal checkpoint=/home/camilo/repos/IsaacGym_repo/IsaacGymEnvs/isaacgymenvs/runs/Anymal_from_omni/anymal.pth test=True num_envs=6 rl_device=cpu sim_device=cpu pipeline=cpu
 ```

 <img src="Images/IsaacGym_baseRepo_load_OmniModel_anymal.png" width=700>

 2) Random location, and no locomotion controller. (basic_ant_test1.py) [^1]

  <img src="Images/IsaacGym_basic_ant_random_location.png" width=700>

 3) **legged_gym project** some changes has been done to force the CPU execution. unfortunately, It wasn't able to upload the controller Omni-model (i.e. policy.pt / anymalTerrain.pth).   [^2]


 ### Notes
 ---
 1) Some testing and commands are listed in the "notes" file  


 ### Reference
 [^1]: [IsaacGym](https://github.com/isaac-sim/IsaacGymEnvs).

 [^2]: [legged_gym](https://github.com/leggedrobotics/rsl_rl).

 [^3]: [OmniIsaacGymEnvs](https://github.com/boredengineering/OmniIsaacGymEnvs). [^3]