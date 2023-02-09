import numpy as np
import rvo2
from coll_gym.envs.methods.method import Method
from coll_gym.envs.utils.action import ActionXY


class ORCA(Method):
    def __init__(self):
       
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def configure(self, config): 
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        
        #Python-RVO2的API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx.
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.pedestrian_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for pedestrian_state in state.pedestrian_states:
                self.sim.addAgent(pedestrian_state.position, *params, pedestrian_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, pedestrian_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, pedestrian_state in enumerate(state.pedestrian_states):
                self.sim.setAgentPosition(i + 1, pedestrian_state.position)
                self.sim.setAgentVelocity(i + 1, pedestrian_state.velocity)

        # 设置最大速度,最大速度是优先速度
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, pedestrian_state in enumerate(state.pedestrian_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
