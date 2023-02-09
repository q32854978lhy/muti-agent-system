import logging

import matplotlib.lines as mlines
import numpy as np
from matplotlib import patches
from numpy.linalg import norm

import gym
import rvo2
from coll_gym.envs.utils.agent import Pedestrian
from coll_gym.envs.utils.info import *
from coll_gym.envs.utils.utils import point_to_segment_dist


class PedestrianRich(gym.Env):
    metadata = {'render.modes': ['pedestrian']}

    def __init__(self):

        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.pedestrians = None
        self.global_time = None
        self.pedestrian_times = None
        
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.pedestrian_num = None
        
        self.states = None
        self.action_values = None
        self.attention_weights = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('pedestrians', 'method') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.pedestrian_num = config.getint('sim', 'pedestrian_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('pedestrian number: {}'.format(self.pedestrian_num))
        if self.randomize_attributes:
            logging.info("Pedestrian's radius and preferred speed: random")
        else:
            logging.info("Pedestrian's radius and preferred speed: default")
        logging.info('The training rule : {}, The test rule : {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square range: {}, circle range: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_random_pedestrian_position(self, pedestrian_num, rule):
        
        if rule == 'square_crossing':
            self.pedestrians = []
            for i in range(pedestrian_num):
                self.pedestrians.append(self.generate_square_crossing_pedestrian())
        elif rule == 'circle_crossing':
            self.pedestrians = []
            for i in range(pedestrian_num):
                self.pedestrians.append(self.generate_circle_crossing_pedestrian())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_pedestrian(self):
        pedestrian = Pedestrian(self.config, 'pedestrians')
        if self.randomize_attributes:
            pedestrian.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # 添加一些噪声以模拟机器人可能与行人的所有可能情况
            px_noise = (np.random.random() - 0.5) * pedestrian.v_pref
            py_noise = (np.random.random() - 0.5) * pedestrian.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.pedestrians:
                min_dist = pedestrian.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        pedestrian.set(px, py, -px, -py, 0, 0, 0)
        return pedestrian

    def generate_square_crossing_pedestrian(self):
        pedestrian = Pedestrian(self.config, 'pedestrians')
        if self.randomize_attributes:
            pedestrian.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.pedestrians:
                if norm((px - agent.px, py - agent.py)) < pedestrian.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.pedestrians:
                if norm((gx - agent.gx, gy - agent.gy)) < pedestrian.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        pedestrian.set(px, py, gx, gy, 0, 0, 0)
        return pedestrian

    

    def reset(self, phase='test', test_case=None):
        
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.pedestrian_times = [0] * self.pedestrian_num
        else:
            self.pedestrian_times = [0] * (self.pedestrian_num if self.robot.method.multiagent_training else 1)
        if not self.robot.method.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('pedestrians', 'method') == 'trajectorynet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    pedestrian_num = self.pedestrian_num if self.robot.method.multiagent_training else 1
                    self.generate_random_pedestrian_position(pedestrian_num=pedestrian_num, rule=self.train_val_sim)
                else:
                    self.generate_random_pedestrian_position(pedestrian_num=self.pedestrian_num, rule=self.test_sim)
                # case0~case[size]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    self.pedestrian_num = 3
                    self.pedestrians = [Pedestrian(self.config, 'pedestrians') for _ in range(self.pedestrian_num)]
                    self.pedestrians[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.pedestrians[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.pedestrians[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.pedestrians:
            agent.time_step = self.time_step
            agent.method.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.method, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.method, 'get_attention_weights'):
            self.attention_weights = list()

        # 获取观察的状态
        if self.robot.sensor == 'coordinates':
            ob = [pedestrian.get_observable_state() for pedestrian in self.pedestrians]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        #计算所有智能体的动作，检测碰撞，更新环境并返回
        pedestrian_actions = []
        for pedestrian in self.pedestrians:
            
            ob = [other_pedestrian.get_observable_state() for other_pedestrian in self.pedestrians if other_pedestrian != pedestrian]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            pedestrian_actions.append(pedestrian.act(ob))

        # 碰撞检测
        dmin = float('inf')
        collision = False
        for i, pedestrian in enumerate(self.pedestrians):
            px = pedestrian.px - self.robot.px
            py = pedestrian.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = pedestrian.vx - action.vx
                vy = pedestrian.vy - action.vy
            else:
                vx = pedestrian.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = pedestrian.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # 机器人和行人的最近距离
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - pedestrian.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # 行人之间的碰撞检测
        pedestrian_num = len(self.pedestrians)
        for i in range(pedestrian_num):
            for j in range(i + 1, pedestrian_num):
                dx = self.pedestrians[i].px - self.pedestrians[j].px
                dy = self.pedestrians[i].py - self.pedestrians[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.pedestrians[i].radius - self.pedestrians[j].radius
                if dist < 0:
                    logging.debug('Collision happens between pedestrians in step()')

        # 检测是否到达目标点
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # 存储状态，动作值和注意权重
            self.states.append([self.robot.get_full_state(), [pedestrian.get_full_state() for pedestrian in self.pedestrians]])
            if hasattr(self.robot.method, 'action_values'):
                self.action_values.append(self.robot.method.action_values)
            if hasattr(self.robot.method, 'get_attention_weights'):
                self.attention_weights.append(self.robot.method.get_attention_weights())

            # 更新所有智能体
            self.robot.step(action)
            for i, pedestrian_action in enumerate(pedestrian_actions):
                self.pedestrians[i].step(pedestrian_action)
            self.global_time += self.time_step
            for i, pedestrian in enumerate(self.pedestrians):
                # 记录行人第一次到达目标的时间
                if self.pedestrian_times[i] == 0 and pedestrian.reached_destination():
                    self.pedestrian_times[i] = self.global_time

            # 计算观察值
            if self.robot.sensor == 'coordinates':
                ob = [pedestrian.get_observable_state() for pedestrian in self.pedestrians]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [pedestrian.get_next_observable_state(action) for pedestrian, action in zip(self.pedestrians, pedestrian_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def render(self, mode='pedestrian', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'pedestrian':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            for pedestrian in self.pedestrians:
                pedestrian_circle = plt.Circle(pedestrian.get_position(), pedestrian.radius, fill=False, color='b')
                ax.add_artist(pedestrian_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'trajectory':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            pedestrian_positions = [[self.states[i][1][j].position for j in range(len(self.pedestrians))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    pedestrians = [plt.Circle(pedestrian_positions[k][i], self.pedestrians[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.pedestrians))]
                    ax.add_artist(robot)
                    for pedestrian in pedestrians:
                        ax.add_artist(pedestrian)
                #添加时间注释
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = pedestrians + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.pedestrian_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    pedestrian_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.pedestrian_num)]
                    ax.add_artist(nav_direction)
                    for pedestrian_direction in pedestrian_directions:
                        ax.add_artist(pedestrian_direction)
            time = plt.text(-2, -5, 'finish time: {}'.format(global_time), fontsize=16)
            ax.add_artist(time)            
            plt.legend([robot], ['Robot'],markerfirst=False,fontsize=16,loc='upper left')
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # 添加机器人及其目标
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16,loc='upper left')

            # 添加行人及其人数
            pedestrian_positions = [[state[1][j].position for j in range(len(self.pedestrians))] for state in self.states]
            pedestrians = [plt.Circle(pedestrian_positions[0][i], self.pedestrians[i].radius, fill=False)
                      for i in range(len(self.pedestrians))]
            pedestrian_numbers = [plt.text(pedestrians[i].center[0] - x_offset, pedestrians[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.pedestrians))]
            for i, pedestrian in enumerate(pedestrians):
                ax.add_artist(pedestrian)
                ax.add_artist(pedestrian_numbers[i])

            # 添加时间注释
            time = plt.text(-1, -5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # 计算注意力分数
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(1, 5 - 0.5 * i, 'Pedestrian {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.pedestrians))]

            # 在每个步骤中计算方向并使用箭头显示方向
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.pedestrian_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, pedestrian in enumerate(pedestrians):
                    pedestrian.center = pedestrian_positions[frame_num][i]
                    pedestrian_numbers[i].set_position((pedestrian.center[0] - x_offset, pedestrian.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        pedestrian.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('pedestrian {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))


                time.set_text('time: {:.2f}'.format(frame_num * self.time_step))

            
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
