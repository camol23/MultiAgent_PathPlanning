''''

    first appoach:
        Substitute gamepad commands 

'''

from MPC_Controller.utils import GaitType, FSM_StateName

# class GaitType(Enum):
#     TROT = 0
#     WALK = 6
#     BOUND = 1


class movement_bridge:
    def __init__(self):

        self.mode_list = [FSM_StateName.RECOVERY_STAND, FSM_StateName.LOCOMOTION]
        self.gait_list = [x for x in GaitType]

        self.gait = self.gait_list[0]
        self.mode = self.mode_list[0]

        self.vx = 0. 
        self.vy = 0.
        self.wz = 0.

        self.aux_counter = 0
        self.aux_idx = 0
            


    def initilize(self, vx = 0.61035156, vy = 0, wz = 0):
        '''
            Define default values 
        '''
        self.vx = vx 
        self.vy = vy
        self.wz = wz

        print("Brindge Settled values ")
        print("Vels. = ", self.vx, self.vy)
        print("Heading = ", self.wz)
        print("Gait init = ", self.gait_list)
        print("Mode init = ", self.mode_list)

    
    def move_FB_cte(self, direction=True, vel=0.9): # vel = 0.6 good try
        '''
            Move along X-axis with constant speed
        '''
 
        self.vy = 0
        self.wz = 0

        self.mode = self.mode_list[1]

        if not direction:
            self.vx = -vel
        else:
            self.vx = vel

    def move_rotate_cte(self, direction=True, vel=0.7):
        '''
            ROtate with constant speed
        '''

        self.vx = 0
        self.vy = 0        

        self.mode = self.mode_list[1]

        if not direction:
            self.wz = -vel
        else:
            self.wz = vel

    def move_stop(self, direction=False):
        self.vx = 0
        self.vy = 0
        self.wz = 0

        self.gait = self.gait_list[0]
        self.mode = self.mode_list[0]


    
    def move_list_sequence(self, jump_count=600):
        '''
            Define a sequence base on basic movements stored in a list
                (*) Constan time to performe each move
        '''
        
        movements = [
            self.move_rotate_cte,
            self.move_FB_cte,
            self.move_FB_cte,
            self.move_stop,
            self.move_rotate_cte
        ]
        
        direction_list = [0, 1, 0, 0, 0]
        
        if (self.aux_counter == 0) or (self.aux_counter >= jump_count):
            
            # print(" LEN = ", (len(movements)-1), self.aux_idx)
            if not (self.aux_idx > (len(movements)-1)): 
                movements[self.aux_idx]( direction_list[self.aux_idx] )
                self.aux_idx += 1
            else:
                self.move_stop() 
                # self.aux_idx = 0  # repeat the sequence

            self.aux_counter = 1
            

        self.aux_counter += 1
        # print("Counter to jump = ", self.aux_counter)