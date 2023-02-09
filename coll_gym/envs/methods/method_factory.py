from coll_avo.methods.attention_rl import Attention_RL
from coll_avo.methods.cadrl import CADRL
from coll_avo.methods.lstm_rl import LstmRL
from coll_gym.envs.methods.linear import Linear
from coll_gym.envs.methods.orca import ORCA


def none_method():
    return None


method_factory = dict()
method_factory['linear'] = Linear
method_factory['orca'] = ORCA
method_factory['none'] = none_method

method_factory['cadrl'] = CADRL
method_factory['lstm_rl'] = LstmRL
method_factory['attention_rl'] = Attention_RL
