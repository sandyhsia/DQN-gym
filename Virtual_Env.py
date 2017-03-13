import time
import pygame
import sys 
import numpy as np
import math
import random as random
from pygame.locals import *
import cv2
import time
from Constant import car_reset_pt

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'DEMO'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 1 # The number of experiment test every 100 episode
color = 125,100, 210
action_space = ['w', 'a', 'd']
default_bounding_lines = [[1., 0., -640.],
                                [1., 0., 0.],
                                [0., 1., -480.],
                                [0., 1., 0.,]]

#border = [[20, 30], [15, 30],
#               [15, 30], [50, 40],
#              [50, 40], [20, 30]]
border= [[]]

limited_eyesight = 1
eyesight = 50

with_cam = 0
speed_level0 = 5
speed_level1 = 10
turn_level0 = math.pi/4
turn_level1 = math.pi/4
car_body_lenth = 10

default_car_center = [100., 20.]
default_angle = 0.
DEBUG_MODE = 0


class Virtual_Env():
        # Virtural Env + Actual Env (captured from camera)
        def __init__(self, ENV_NAME, w, h): 
                # maybe tell me how many cars in gray_dst some time later
                # might related to observation space

                pygame.init()
                # print "w, h:", w, h
                self.screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption(ENV_NAME)
                pygame.key.set_repeat(50)
                self.screen.fill(color)

                self.action_space = action_space
                self.observation_space = np.zeros(360)
                self.bounding_lines = default_bounding_lines # default
                self.bounding_lines[0][2] = -w
                self.bounding_lines[2][2] = -h
                # self.bounding_cnt = np.array([border[0], border[2], border[4]], dtype=np.int)
                self._with_cam = with_cam
                self.w = w
                self.h = h
                self.default_car_center = [100., 20.]
                self.default_angle = 0.
                self.car_target_pt = [0., 0.]


                for i in range(len(border)/2):
                        if border[i*2+0][1] == border[i*2+1][1]:
                                self.bounding_lines.append([0., 1., -border[i*2+0][1]])

                        elif border[i*2+0][0] == border[i*2+1][0]:
                                self.bounding_lines.append([1., 0., -border[i*2+0][0]])

                        else:
                                arr1 = [[border[i*2+0][1], 1],
                                [border[i*2+1][1], 1]]
                                arr2 = [-border[i*2+0][0], -border[i*2+1][0]]
                                border_line = np.linalg.solve(arr1, arr2)
                                self.bounding_lines.append([1., border_line[0], border_line[1]])


        def reset(self):
                self.car_center = [int(random.random()*(self.w- 2*car_body_lenth)+car_body_lenth), int(random.random()*(self.h - 2*car_body_lenth)+car_body_lenth)]
                self.car_target_pt = [int(random.random()*(self.w- 2*car_body_lenth)+car_body_lenth), int(random.random()*(self.h- 2*car_body_lenth)+car_body_lenth)]
                '''
                while cv2.pointPolygonTest(self.bounding_cnt, (self.car_center[0], self.car_center[1]), False) >= 0 or cv2.pointPolygonTest(self.bounding_cnt, (self.car_target_pt[0], self.car_target_pt[1]), False) >= 0:
                        self.car_center = [int(random.random()*self.w), int(random.random()*self.h)]
                        self.car_target_pt = [int(random.random()*self.w), int(random.random()*self.h0)]
                '''

                # print self.car_center
                self.angle = round(random.random()*360 - 180)
                # self.angle = 180
                print "reset car location:", self.car_center, self.angle
                print "reset car target location:", self.car_target_pt
                state, reward, done = self.step(self.car_center, self.angle, 0)
                return state

        def step(self, car_center, angle, action):

                self.screen.fill(color)
                center = [0., 0.]
                center = car_center
                if self._with_cam == 0:
                        car_center, angle, done = self.agent_action(car_center, angle, action)
                        # print car_center, angle, done
                        self.car_center = car_center
                        self.angle = angle
                        # print car_center, angle, done
                else:
                        pass

                if self._with_cam == 0 and done == True:
                        #pygame.draw.line(self.screen, [0,0,0], border[0], border[1], 3)
                        #pygame.draw.line(self.screen, [0,0,0], border[2], border[3], 3)
                        #pygame.draw.line(self.screen, [0,0,0], border[4], border[5], 3)
                        pygame.display.update()
                        return np.zeros(362), -10, done

                else:
                        #pygame.draw.line(self.screen, [0,0,0], border[0], border[1], 3)
                         #pygame.draw.line(self.screen, [0,0,0], border[2], border[3], 3)
                         #pygame.draw.line(self.screen, [0,0,0], border[4], border[5], 3)
                        pygame.display.update()

                        solution = np.zeros((360, 2))
                        intersect_points_vec = np.zeros((360,2))
                        distance_vec = np.zeros(360)
                        bounding_lines = self.bounding_lines

                        k_arr = np.zeros(360)
                        line_param_arr = np.ones((360, 2))
                        c = np.zeros(360)
                        for i in range(360):
                                if i != 180 and i != 0:
                                        k_arr[i] = math.tan((i-180)*(math.pi)/180)
                                        line_param_arr[i] = (1.00, -1/k_arr[i])
                                        c[i] = np.dot(center, line_param_arr[i].T)
                                else:
                                        k_arr[i] = 0
                                        line_param_arr[i] = (0.00, 1.00)
                                        c[i] = np.dot(center, line_param_arr[i].T)


                        for j in range(len(bounding_lines)):
                                for i in range(360):
                                        param = np.zeros((2, 2))
                                        param[0] = line_param_arr[i]
                                        param[1] = (bounding_lines[j][0], bounding_lines[j][1])
                                        bias = (c[i], -bounding_lines[j][2])
                                        if param[0][0]*param[1][1] != param[0][1] * param[1][0]:
                                                solution[i] = np.linalg.solve(param, bias)
                                        if(solution[i][0] >=self.w):
                                                solution[i][0] = self.w
                                        if(solution[i][0] <=0):
                                                solution[i][0] =0
                                        if(solution[i][1] >= self.h):
                                                solution[i][1] = self.h
                                        if(solution[i][1] <= 0):
                                                solution[i][1] = 0
                                
                                if j <= 3:
                                        for i in range(360):
                                                if int(vector_direction(center, solution[i])) == (i-180)*(-1):
                                                        if intersect_points_vec[i][0] == 0 and intersect_points_vec[i][1] == 0:
                                                                intersect_points_vec [i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])
                                                        elif two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])

        
                                else:
                                        for i in range(360):
                                                if int(vector_direction(center, solution[i])) == (i-180)*(-1) and abs(vector_direction(border[(j-4)*2+0], solution[i]) - vector_direction(border[(j-4)*2+1], solution[i])) == 180:
                                                        if two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])


                        if limited_eyesight == 1:
                                for i in range(360):
                                        if distance_vec[i] > eyesight:
                                                intersect_points_vec[i] = (center[0] + eyesight*np.cos((-i+180)*math.pi/180), center[1] - eyesight*np.sin((-i+180)*math.pi/180))
                                                distance_vec[i] = eyesight


                        #for i in range(360):
                        #       if(i%90== 0):
                        #                pygame.draw.line(self.screen, [255,0,255], center, intersect_points_vec[i], 1)
                        pygame.display.update()



                        ''' Give next_state, reward, done 
                            to agent '''
                        next_distance_vec = distance_vec
                        for i in range(360):
                                next_distance_vec[i] = distance_vec[(180-int(angle)+i)%360]

                        next_distance_vec = np.append(next_distance_vec, two_point_distance(center, self.car_target_pt))
                        
                        target_direction = vector_direction(center, self.car_target_pt)
                        if angle >= 0 and angle < 180:
                                angle_to_append = angle
                        elif angle >= 180 and angle <=360:
                                angle_to_append = angle - 360

                        if target_direction - angle_to_append >180:
                                angle_diff = 360 - (target_direction - angle_to_append)
                        elif target_direction - angle_to_append < -180:
                                angle_diff = 360 +(target_direction - angle_to_append)
                        else:
                                angle_diff = target_direction-angle_to_append
                        next_distance_vec = np.append(next_distance_vec, (angle_diff))

                        head = [0., 0.]
                        angle_in_rad = angle*math.pi/180
                        head[0] = center[0] + car_body_lenth*np.cos(angle_in_rad)
                        head[1] = center[1] - car_body_lenth*np.sin(angle_in_rad)
                        if two_point_distance(head, self.car_target_pt) <= 10 or two_point_distance(center, self.car_target_pt) <= 10:
                                done = True
                                reward = 5
                                print "Agent: Yoho!!"

                        if done == False:
                                reward = self.reward_method(next_distance_vec, action)

                        if DEBUG_MODE == 1:
                                print "next_distance_vec[0:10]",  next_distance_vec[0:10]
                                print "target direction", vector_direction(center, self.car_target_pt), "angle", angle
                                print "dis:", next_distance_vec[360], "angle diff:", next_distance_vec[361] 
                                print "reward", reward
                                time.sleep(1)


                        return next_distance_vec, reward, done

        def agent_action(self, car_center, angle, action):

                head = [0., 0.]
                tail = [0., 0.]
                angle_in_rad = angle*math.pi/180
                center = car_center
                
                if action == 0: #w
                        if action == 0:
                                center[0] += speed_level0 *np.cos(angle_in_rad)
                                center[1] -= speed_level0*np.sin(angle_in_rad)
                        else:
                                center[0] += speed_level1 *np.cos(angle_in_rad)
                                center[1] -= speed_level1*np.sin(angle_in_rad)

                if action == 1: #a
                        if action == 1:
                                angle_in_rad += turn_level0

                        else:
                                angle_in_rad+= turn_level1

                if action == 30: # s, cancel
                        if action == 3:
                                center[0] -= speed_level0 *np.cos(angle_in_rad)
                                center[1] += speed_level0*np.sin(angle_in_rad)

                        else:
                                center[0] -= speed_level1 *np.cos(angle_in_rad)
                                center[1] += speed_level1*np.sin(angle_in_rad)
                
                if action == 2: # d
                        if action == 2:
                                angle_in_rad -= turn_level0

                        else:
                                angle_in_rad -= turn_level1

                else:
                        center = car_center

                head[0] = center[0] + car_body_lenth*np.cos(angle_in_rad)
                head[1] = center[1] - car_body_lenth*np.sin(angle_in_rad)
                tail[0] = center[0] + car_body_lenth*np.cos(angle_in_rad + math.pi)
                tail[1] = center[1] - car_body_lenth*np.sin(angle_in_rad + math.pi)
      
                pygame.draw.line(self.screen, [255,0,0], head, tail, 3)
                pygame.draw.circle(self.screen, [0,0,0], (int(head[0]),int(head[1])), 2)
                pygame.draw.circle(self.screen, [0,0,0], (int(self.car_target_pt[0]),int(self.car_target_pt[1])), 2)
                pygame.display.update()


                return_angle = int(180*angle_in_rad/math.pi)%360
                return_car_center = car_center
                
                if head[0] <= self.w and head[0] >= 0 and head[1] <= self.h and head[1] >= 0 and tail[0] <= self.w and tail[0] >= 0 and tail[1] <= self.h and tail[1] >=0:
                        done = False
                else:
                        done = True
                        print "out of playground."

                '''if cv2.pointPolygonTest(self.bounding_cnt, (head[0], head[1]), False) >= 0:
                        done = True
                        print "Head! Boom!"
                '''

                '''if cv2.pointPolygonTest(self.bounding_cnt, (tail[0], tail[1]), False) >= 0:
                        done = True
                        print "Tail! Boom!"
                '''

                return car_center, return_angle, done

        def reward_method(self, next_state, this_state_action):

                if next_state[360] >0:
                        reward = 10/next_state[360]
                else:
                        reward = 10

                if next_state[361] != 0:
                        reward += 10/(abs(next_state[361]))
                else:
                        reward += 10

                space = 0
                action = this_state_action
                
                if self._with_cam == 0:
                        '''
                        for i in range(len(next_state)):
                                space += next_state[i]

                        reward += 8*space/(360*eyesight) - 7
                        '''
                        ''' Calculation on free space, 
                            if the next_distance_vec is half space, then reward is 0'''

                        if action == 0:      # w0
                                reward += 0.1
                        elif action == 1:   # w1
                                reward += 0.00
                        elif action == 2 or action == 6: # a0, d0
                                reward += 0.000
                        elif action == 3 or action == 7: # a1, d1
                                reward += 0.000
                        elif action == 4 or action == 5: # s0, s1
                                reward += -0.000
                        else:
                                pass
                        '''
                        print "next_state", next_state[0:10]
                        print "space reward: ", 8*space/(360*eyesight) - 7
                        print "this state action: ", this_state_action
                        print "reward", reward
                        time.sleep(1)
                        '''

                return reward

def two_point_distance(start_pt, end_pt):
        return math.sqrt((start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2)

def vector_direction(start_pt, end_pt):
        start_pt = axis_convert2_normal(start_pt)
        end_pt = axis_convert2_normal(end_pt)
        pi = math.pi
        angle_in_rad = math.atan2((end_pt[1] - start_pt[1]), (end_pt[0] - start_pt[0]))
        angle_in_degree = (angle_in_rad/math.pi)*180
        return round(angle_in_degree)

def axis_convert2_normal(point_xy_in_video):
        return (point_xy_in_video[0], -point_xy_in_video[1])