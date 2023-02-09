from gym.envs.registration import register

register(
    id='PedestrianRich-v2020',
    entry_point='coll_gym.envs:PedestrianRich',
)
