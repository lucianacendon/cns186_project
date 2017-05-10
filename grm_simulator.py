"""
    Caltech CNS186 - Final Project
    Implementation of Motion-based Navigation Algorithm Based on Generalized Regressive Motion 
    Written by: Luciana H.P.Cendon
"""

# User-Defined Parameters
VERBOSE = False
SCREEN_SIZE = (600, 400)
FLY_SIZE = 30           # Each fly is FLY_SIZE x FLY_SIZE (in pixel units)
N_FLYES = 10
CVA = 30               # In degrees
RADIUS_T_GRM = 100     # In pixels
PACMAN_MODE = True
SAMPL_INTERVAL = 10    # Sampling interval in number of iterations
SPEED_MIN = 0.3        # The speed magnitude is generated randomly from this interval. This variable is not completely accurate 
                       # since the actual speed interval shown on the screen does vary with the number of flies inside the arena: 
                       # the higher the number of flies, the faster they move and the smallest these numbers needs to be set. These
                       # speed limits are therefore used in a symbolic way
SPEED_MAX = 0.5         
COLL_AVOIDANCE = True   # This activates the algorithm or not
SIM_DURATION = 600      # Simulation time in seconds

background_image = "./images/tangerine.jpg"
fly_image = "./images/fly.png"


import pygame
from pygame.locals import *
from sys import exit
import random
import math
import numpy as np
import matplotlib.pyplot as plt


class Entity():    
    def __init__(self, arena, tag, image):
        self.arena = arena
        self.image = image
        self.collision_rad = 0.25*FLY_SIZE
        self.position = np.zeros([1,2])
        self.tag = tag   
        self.id = 0      # This id is used to identify the object among all the objects


    def render(self, surface):
    	"""
        	This function simply blits the entities image to the screen. It first adjusts the coordinates 
        	so that the current location is under the center of the image rather than the top left. 
    	"""       
        x = self.position[0][0]
        y = self.position[0][1]
        w, h = self.image.get_size()
        surface.blit(self.image, (x-w, y-h))   # Make sure the fly does not go out of the screen

        if (VERBOSE == True):
            font = pygame.font.SysFont(None, 20)
            text = font.render("Fly "+str(self.id), True, (0,0,255))
            surface.blit(text, (x, y))



class Fly(Entity):

    def __init__(self, arena, image):     
        Entity.__init__(self, arena, "fly", image)  # Calls the constructor for the "Entity" base class

        # Coordinate variables
        self.speed = np.zeros([1,2])
        self.mag_speed = 0
        self.dist_travelled = 0 
        self.orientation = np.zeros([1,2])
        self.left_eye_axis = np.zeros([1,2])
        self.right_eye_axis = np.zeros([1,2])
        self.angle = 0           # Stores angle in degrees w.r.t x-axis: positive if heading towards cartesian 3Q or 4Q 
                                 # and negative if heading toward 1Q or 2Q    

        # Useful list variables
        self.stored_angles = {}  # This list will store the angles between fly orientation and fly_in_the_arena. Each angle will be indexed 
        					     # by fly_id. It will be useful in detecting GRM      					     
        self.potential_targets = {}  # This list will store the id's of flies with which the current fly has detected GRM. Those flies
        							 # usually will detect GRM from the current fly at the same time, and in this case both of them will 
                                     # stop. Their ids need to be stored in order to make sure that they DONT stop while the current fly 
                                     # is stopped. 

        # Boolean Variables used for switching states
        self.collided = False
        self.detected_GRM = False
        self.no_stop = False

        # Useful for analysis
        self.n_stops = 0
        self.n_collisions = 0

        # Starts state machine for the fly
        self.statemachine = StateMachine()  

        # Create instances of each state and initializes them.
        # This also passes the fly iself to these states so they 
		# can access its respective indicator boolean variables                                                 
        start_moving = Start_Moving(self)
        moving = Moving_State(self)                
        waiting = Waiting_State(self)
        stunned = Stunned_State(self)

        # Add both states to the state machine   
        self.statemachine.add_state(start_moving)
        self.statemachine.add_state(waiting)
        self.statemachine.add_state(moving)  
        self.statemachine.add_state(stunned)           


    def render(self, surface):       
        Entity.render(self, surface)


    def process(self, time_passed_seconds):
        if (VERBOSE == True):
                print ('')
                print ('Checking Field of Vision of Fly '+str(self.id))


        if COLL_AVOIDANCE == True:
            # Scanning all flies in the arena
            for fly_in_arena in self.arena.entities.values():
                if fly_in_arena.id != self.id:
                   
                    # Finding the orientation of LEFT eye. I'm just applying rotation matrix to the orientation of the fly
                    rot_angle = -0.25 * math.pi  
                    self.left_eye_axis[0][0] = self.orientation[0][0] * math.cos(rot_angle) - self.orientation[0][1] * math.sin(rot_angle)
                    self.left_eye_axis[0][1] = self.orientation[0][0] * math.sin(rot_angle) + self.orientation[0][1] * math.cos(rot_angle)

                    # Finding the orientation of RIGHT eye. I'm just applying rotation matrix to the orientation
                    rot_angle = 0.25 * math.pi  
                    self.right_eye_axis[0][0] = self.orientation[0][0] * math.cos(rot_angle) - self.orientation[0][1] * math.sin(rot_angle)
                    self.right_eye_axis[0][1] = self.orientation[0][0] * math.sin(rot_angle) + self.orientation[0][1] * math.cos(rot_angle)
                    
                    Fly.see_left(self, fly_in_arena)
                    Fly.see_right(self, fly_in_arena)
                    Fly.collision_detector(self , fly_in_arena)
                    self.statemachine.decide(time_passed_seconds)

        if COLL_AVOIDANCE == False:
            for fly_in_arena in self.arena.entities.values():
                if fly_in_arena.id != self.id:
                    Fly.collision_detector(self , fly_in_arena)
                    self.statemachine.decide(time_passed_seconds)


    def see_left(self, fly_in_arena):
        """
            Checks whether 'fly_in_arena' is in the left field of vision of the fly
        """
        CVA_rad = np.deg2rad(CVA)
        v_prime = vec_from_points(self.position , fly_in_arena.position)  # v_prime is a vector from the current fly to the fly being analyzed
        dot_product = np.dot(self.left_eye_axis, v_prime)
        mag_vprime = np.linalg.norm(v_prime)
        angle_d = math.acos(dot_product/mag_vprime) 

        if(VERBOSE == True):
            print('Distance between fly '+str(self.id)+' LEFT EYE and fly '+str(fly_in_arena.id)+': '+str(mag_vprime))
            print ('Angle between fly '+str(self.id)+' and fly '+str(fly_in_arena.id)+': '+str(angle_d))

        if (angle_d < (0.25*math.pi + CVA_rad)) and mag_vprime < RADIUS_T_GRM:
            Fly.grm_detector(self , fly_in_arena)   # Checks for GRM only in case the fly is seen  

            if (VERBOSE == True):  
                print('Fly '+str(fly_in_arena.id)+' seen on the LEFT of fly '+str(self.id))
            return
        else:
            if (VERBOSE == True):
                print('No flies in the LEFT vision field')
            return 


    def see_right(self, fly_in_arena):
        """
            Checks whether 'fly_in_arena' is in the right field of vision of the fly
        """
        CVA_rad = np.deg2rad(CVA)
        v_prime = vec_from_points(self.position , fly_in_arena.position)  # v_prime is a vector from the current fly to the fly being analyzed
        dot_product = np.dot(self.right_eye_axis, v_prime)
        mag_vprime = np.linalg.norm(v_prime)    # magnitude of the vector v_prime
        angle_d = math.acos(dot_product/mag_vprime) 

        if (VERBOSE == True):
            print ('Angle between fly '+str(self.id)+' RIGHT EYE and fly '+str(fly_in_arena.id)+': '+str(angle_d))

        if (angle_d < (0.25*math.pi + CVA_rad)) and mag_vprime < RADIUS_T_GRM:
            Fly.grm_detector(self , fly_in_arena)   # Checks for GRM only in case the fly is seen

            if (VERBOSE == True):
                print('Fly '+str(fly_in_arena.id)+' seen on the RIGHT of fly '+str(self.id))
            return 
        else:
            if (VERBOSE == True):
                print('No flies in the RIGHT vision field')
            return 


    def collision_detector(self, fly_in_arena):
        """
            This function detects collisions between two entities
        """
        # Two circles intersect if the distance between their centers is between the sum 
        # and the difference of their radii.
        min_bound = np.power(self.collision_rad - fly_in_arena.collision_rad , 2)
        max_bound = np.power(self.collision_rad + fly_in_arena.collision_rad , 2)
        dist_centers = np.power(self.position[0][0] - fly_in_arena.position[0][0] , 2) + np.power(self.position[0][1] - fly_in_arena.position[0][1] , 2)

        if dist_centers >= min_bound and dist_centers <= max_bound:
            self.collided = True
            self.n_collisions += 1
            fly_in_arena.n_collisions += 1
            self.arena.n_collisions += 1



    def grm_detector(self, fly_in_arena): 
        # It will store the angle between the orientation of the fly and the vector between position of the fly and the 
        # position of fly_in_arena inside the list called "stored_angle". Each angle will be indexed by fly_id.
        # Then, at each iteration, it checks wether this angle gets smaller. If it does, it means there's GRM.

        v_prime = vec_from_points(self.position , fly_in_arena.position)  # v_prime is a vector from the current fly to the fly being analyzed
        mag_vprime = np.linalg.norm(v_prime)    # Magnitude of the vector v_prime
        dot_product = np.dot(self.orientation, v_prime)
        angle = math.acos(dot_product/(mag_vprime))

        # Adding to the list of current angles
        if fly_in_arena.id in self.stored_angles:
            if abs(angle) < self.stored_angles[fly_in_arena.id]:

                # Only adds fly_in_arena to the potential target list of the current fly is not already in the fly_in_arena list!
                if self.id not in fly_in_arena.potential_targets:
                    self.potential_targets[fly_in_arena.id] = fly_in_arena

                    self.detected_GRM = True
                    fly_in_arena.no_stop = True

                if VERBOSE == True:
                    print('Fly'+str(self.id)+' sees GRM from fly'+str(fly_in_arena.id))

        self.stored_angles[fly_in_arena.id] = abs(angle)
		



# This class will manage the states. It stores an instance of each of the states in a dictionary and manages the currently active state. 
class StateMachine(object):

    def __init__(self):
        self.states = {}            # Stores the states in a list
        self.active_state = None    # The currently active state


    def add_state(self, state):
        # Add a state to the internal dictionary where the states are stored
        self.states[state.name] = state

    # This function runs once per frame, and calls the execute_state on the active state
    def decide(self, time_passed_seconds):
        # Only continue if there is an active state
        if self.active_state is None:
            return

        # Perform the actions of the active state, and check conditions
        self.active_state.execute_state(time_passed_seconds)
        new_state_name = self.active_state.check_conditions()

        if new_state_name is not None:    # If there's a new state, it will perform exit actions 
            self.set_state(new_state_name)

    def set_state(self, new_state_name):
        # Change states and perform any exit / entry actions
        if self.active_state is not None:
            self.active_state.leave_state()                    # If active_state = none, then new_state_name will be set to be the active state.
        self.active_state = self.states[new_state_name]        # It looks in the dictionary structure (self.states) using the new_state_name.
        self.active_state.start_state()                        # Start performing the start_state for that state





# The base State class does not actually do anything other than store the name of the state in the constructor. 
# The remaining functions in State do nothing. We need these empty functions because not all of the states we 
# will be building will implement all of the functions in the base class. 
class State(object):
    def __init__(self, name):
        self.name = name
    def start_state(self):
        pass
    def check_conditions(self):
        pass
    def execute_state(self, time_passed_seconds):
        pass
    def leave_state(self):
        pass





class Start_Moving(State):
    
    def __init__(self, fly):
        
        State.__init__(self, "Start_Moving")   
        self.fly = fly

    def start_state(self):

        self.fly.orientation[0][0] = np.random.uniform(-1,1)
        self.fly.orientation[0][1] = np.random.uniform(-1,1) 

        # Normalizing the vector to make it unit vector (direction of movement)
        mag_orient = np.linalg.norm(self.fly.orientation)
        self.fly.orientation = self.fly.orientation/mag_orient

        # Angle of orientation of the fly with the x-axis
        self.angle = math.atan2(-self.fly.orientation[0][1] , self.fly.orientation[0][0])  # math.atan2(y,x)

        # The image is already flipped 90. Needs to rotate 90 in the clockwise direction (therefore -90)
        self.fly.image = pygame.transform.rotate(self.fly.image, (self.angle - (0.5*math.pi))*57.2958)        

        # 1mm = 3.779 pixels. The user-defined speed limits are multiplied by 37.79 to convert them to pixels/s
        self.fly.mag_speed = np.random.randint(37.79*SPEED_MIN , 37.79*SPEED_MAX) 
        self.fly.speed = self.fly.orientation * self.fly.mag_speed
        


    def execute_state(self, time_passed_seconds):
        w, h = SCREEN_SIZE
        displacement = np.zeros([1,2])

        displacement[0][0] = self.fly.speed[0][0] * time_passed_seconds   # Displacement in the x-direction
        displacement[0][1] = self.fly.speed[0][1] * time_passed_seconds   # Displacement in the y-direction

        self.fly.position[0][0] += displacement[0][0]    # Displacement is given in pixels
        self.fly.position[0][1] += displacement[0][1]

        self.fly.dist_travelled += np.linalg.norm(displacement)   

        if PACMAN_MODE == False:
	        # If the image goes off the end of the screen, invert the speed direction
	        if self.fly.position[0][0] > w or self.fly.position[0][0] < 0 or self.fly.position[0][1] > h or self.fly.position[0][1] < 0:
	            self.fly.speed = -self.fly.speed
	            self.fly.orientation = -self.fly.orientation
	            self.fly.image = pygame.transform.rotate(self.fly.image, 180) 
		
        if PACMAN_MODE == True:
            if self.fly.position[0][0] > w:
                self.fly.position[0][0] = 0
            if self.fly.position[0][0] < 0:
                self.fly.position[0][0] = w
            if self.fly.position[0][1] > h:
                self.fly.position[0][1] = 0
            if self.fly.position[0][1] < 0:
                self.fly.position[0][1] = h    		    


    def check_conditions(self):

        if self.fly.detected_GRM == True and self.fly.no_stop == False:
        	self.fly.n_stops += 1
        	self.fly.arena.n_stops += 1
        	return "Waiting"

        if self.fly.collided == True:
            return "Stunned"





class Waiting_State(State):
    
    def __init__(self, fly):
        
        State.__init__(self, "Waiting")
        self.fly = fly

    # Theres nothing to do in the waiting state except wait for the condition to be met. 
    # So no need to write execute_state for this state

    def check_conditions(self):

        indicator = random.randint(0,20)
        if indicator == 1:
          	return "Moving"


    def leave_state(self):
    	# Re-setting indicator variables
    	self.fly.detected_GRM = False
        self.fly.collided = False

        for target in self.fly.potential_targets.values(): 
            target.no_stop = False

        self.fly.potential_targets.clear()
        self.fly.stored_angles.clear()





class Moving_State (State):
    
    def __init__(self, fly):
        
        State.__init__(self, "Moving")   
        self.fly = fly

   
    def execute_state(self, time_passed_seconds):
        w, h = SCREEN_SIZE
        displacement = np.zeros([1,2])

        displacement[0][0] = self.fly.speed[0][0] * time_passed_seconds   # Displacement in the x-direction
        displacement[0][1] = self.fly.speed[0][1] * time_passed_seconds   # Displacement in the y-direction

        self.fly.position[0][0] += displacement[0][0]    # Displacement is given in pixels
        self.fly.position[0][1] += displacement[0][1]

        self.fly.dist_travelled += np.linalg.norm(displacement)   

        if PACMAN_MODE == False:
	        # If the image goes off the end of the screen, invert the speed direction
	        if self.fly.position[0][0] > w or self.fly.position[0][0] < 0 or self.fly.position[0][1] > h or self.fly.position[0][1] < 0:
	            self.fly.speed = -self.fly.speed
	            self.fly.orientation = -self.fly.orientation
	            self.fly.image = pygame.transform.rotate(self.fly.image, 180) 
		
        if PACMAN_MODE == True:
            if self.fly.position[0][0] > w:
                self.fly.position[0][0] = 0
            if self.fly.position[0][0] < 0:
                self.fly.position[0][0] = w
            if self.fly.position[0][1] > h:
                self.fly.position[0][1] = 0
            if self.fly.position[0][1] < 0:
                self.fly.position[0][1] = h   

    def check_conditions(self):

        if self.fly.detected_GRM == True and self.fly.no_stop == False:
        	self.fly.n_stops += 1
        	self.fly.arena.n_stops += 1
        	return "Waiting"

        if self.fly.collided == True:

            return "Stunned"

    



class Stunned_State(State):
    
    def __init__(self, fly):
        
        State.__init__(self, "Stunned")
        self.fly = fly

    def start_state(self):
        self.fly.speed = -self.fly.speed
        self.fly.orientation = -self.fly.orientation
        self.fly.position += 30 * self.fly.orientation
        self.fly.image = pygame.transform.rotate(self.fly.image, 180) 

        for target in self.fly.potential_targets.values():
            target.position += 30 * target.orientation
            target.no_stop = False    

            target.statemachine.set_state("Moving")


    def check_conditions(self):

        indicator = random.randint(0,10)
        if indicator == 1:
            return "Moving"


    def leave_state(self):
        # Re-setting indicator variables
        self.fly.detected_GRM = False
        self.fly.collided = False

        self.fly.potential_targets.clear()
        self.fly.stored_angles.clear()





class Arena(object):
    def __init__(self):   

	    self.entities = {}
	    self.entity_id = 0        
	    self.n_collisions = 0
	    self.n_stops = 0
	    self.background = pygame.image.load(background_image).convert()
	    self.background = pygame.transform.scale(self.background, (SCREEN_SIZE))    # scaling the image to fit the screen
  

    def add_entity(self, entity):

        self.entities[self.entity_id] = entity 
        entity.id = self.entity_id   
        self.entity_id += 1


    def process(self, time_passed_seconds):  

    	for fly in self.entities.values(): 
            fly.process(time_passed_seconds)


    def render(self, surface):

        surface.blit(self.background, (0, 0))
        for entity in self.entities.itervalues():
            entity.render(surface)


    def display_information(self,surface): 
        w,h = SCREEN_SIZE
        black = (0,0,0)
        font = pygame.font.SysFont(None, 20)
        text = font.render("No.collisions: "+str(self.n_collisions), True, black)
        surface.blit(text,(w-120,18))
        text = font.render("No.stops: "+str(self.n_stops), True, black)
        surface.blit(text,(w-120,36))
        text = font.render("CVA: "+str(CVA), True, black)
        surface.blit(text,(w-120,0))
        text = font.render("T_grm: "+str(RADIUS_T_GRM), True, black)
        surface.blit(text,(w-120,54))







def main():
    pygame.init()   # This will initialize all pygame modules
    n_iterations = 0
    sim_time = 0
    it_number = -1
    collisions = []
    stops = []

    screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32) # ((width screen, height screen), flags =0 (for no flags),  bit-depth for colors - 32 is the max)
    pygame.display.set_caption("CNS186 - Final Project")

    arena = Arena()
    clock = pygame.time.Clock()
    background = pygame.image.load(background_image).convert() # convert background image to screen format
   

    # Loading fly image and making it 30x30 pixels
    fly_im = pygame.image.load(fly_image)
    fly_im = pygame.transform.scale(fly_im, (FLY_SIZE, FLY_SIZE))
  

    for fly_numb in range(N_FLYES):
       	fly = Fly(arena, fly_im)
       	fly.position[0][0] = random.randint(0, SCREEN_SIZE[0])    # Initialize each fly to a random position
       	fly.position[0][1] = random.randint(0, SCREEN_SIZE[1])

        fly.statemachine.set_state("Start_Moving")   # Make fly move by setting state of each of them to "Moving_State"
        arena.add_entity(fly)                        # Adds moving fly to the arena

    while True:
        for event in pygame.event.get():  # The function pygame.event.get() returns any events waiting for us.  
            if event.type == QUIT:        # If the eve nt is of type QUIT (generated when user clicks close button)
                                          # then we call exit() to shut it down
                display_final_results(arena, n_iterations, collisions, stops, sim_time)                        
                exit()

        if SIM_DURATION != 0:
            if sim_time >= SIM_DURATION:
                display_final_results(arena, n_iterations, collisions, stops, sim_time) 
                exit()

        # Clock tick
        time_passed = clock.tick(30)                 # Game will run at a maximum 30 frames per second (smooth motion)
        time_passed_seconds = time_passed / 1000.0   # From miliseconds to seconds: easier to have it in seconds to compute distances  
        sim_time += time_passed_seconds
        
        # Updating variables and screen
        arena.process(time_passed_seconds)        
        arena.render(screen)
        arena.display_information(screen) 

        pygame.display.update()
        n_iterations += 1
        it_number += 1

        if it_number > SAMPL_INTERVAL:
            collisions.append(arena.n_collisions)
            stops.append(arena.n_stops)
            it_number = 0

    

def vec_from_points(A, B):
    return np.array([B[0][0]-A[0][0] , B[0][1]-A[0][1]])


def display_final_results(arena, n_iterations, collisions, stops, sim_time):
	#"""
	#	This funcion displays the resuls collected during simulation
	#"""
    print('')
    print('')    
    print('Fly Activity Report')
    print('-------------------')
    for fly in arena.entities.values():
        print('Fly'+str(fly.id)+': number of stops = '+str(fly.n_stops)+' , number of collisions = '+str(fly.n_collisions))
        print('Total Distance Travelled:'+str(fly.dist_travelled))
        print ('')

    # Subtracting previous values from each value in "collisions" and "stops" vectors
    corrected_stp = []
    corrected_col = []

    for i in range(len(stops)-1):
        corrected_col.append(collisions[i+1]-collisions[i])
        corrected_stp.append(stops[i+1]-stops[i])

    tot_dist = 0
    # Dividing by the total distance travelled
    for fly in arena.entities.values():
        tot_dist += fly.dist_travelled

    tot_dist_cm = tot_dist/37.79 # Converting total distance to cm

    if COLL_AVOIDANCE == True:
        plt.plot([corrected_stp],[corrected_col],"bo")
        plt.suptitle('Tot. Dist Travelled by the flies: '+str(round(tot_dist_cm,1))+' cm ; Tot. number of collisions: '+str(round(arena.n_collisions,1)), fontsize = 12)
        plt.title('Simulation Time: '+str(sim_time)+' seconds', fontsize = 12)
        plt.xlabel('Number of stops')
        plt.ylabel('Number of Collisions')

        ax = plt.gca()
        cur_ylim = ax.get_ylim(); 
        cur_xlim = ax.get_xlim(); 
        ax.set_xlim([-2,cur_xlim[1]])
        ax.set_ylim([-2,cur_ylim[1]])
        ax.text(1,1, 'Speed range: '+str(SPEED_MIN)+' - '+str(SPEED_MAX), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1,0.95, 'Sampling Interval = '+str(SAMPL_INTERVAL)+' it.', horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1,0.90, 'CVA = '+str(CVA)+' degrees', horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1,0.85, 'Tgrm: '+str(RADIUS_T_GRM)+' pixels', horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1,0.80 , 'N. Flyes: '+str(N_FLYES), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)                
        ax.text(1,0.75, 'CA = '+str(COLL_AVOIDANCE), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        plt.show()

    if COLL_AVOIDANCE == False:
        x_axis = np.linspace(SAMPL_INTERVAL, SAMPL_INTERVAL*len(collisions), len(collisions)) # Syntax: numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]
        plt.plot(x_axis, collisions)
        plt.suptitle('Tot. Dist Travelled by the flies: '+str(round(tot_dist_cm,1))+' cm ; Tot. number of collisions: '+str(round(arena.n_collisions,1)), fontsize = 12)
        plt.title('Simulation Time: '+str(sim_time)+' seconds', fontsize = 12)
        plt.xlabel('Sample Number')
        plt.ylabel('Number of Collisions')

        ax = plt.gca()
        ax.text(1, 0.15, 'N. Flyes: '+str(N_FLYES), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes)
        ax.text(1, 0.1, 'Speed range: '+str(SPEED_MIN)+' - '+str(SPEED_MAX), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes)
        ax.text(1,0.05, 'Sampling Interval = '+str(SAMPL_INTERVAL)+' it.', horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes)
        ax.text(1,0, 'Collision Avoidance = '+str(COLL_AVOIDANCE), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes)
        plt.show()



if __name__ == "__main__":    
    main()




