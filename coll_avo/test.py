import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from coll_avo.utils.experience import Experience
from coll_gym.envs.methods.method_factory import method_factory
from coll_gym.envs.utils.agent import Robot
from coll_gym.envs.methods.orca import ORCA


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--train_parameter', type=str, default='parameters/train.config')
    parser.add_argument('--method', type=str, default='orca')
    parser.add_argument('--method_dir', type=str, default=None)
    parser.add_argument('--imitation_learning', default=False, action='store_true')
    parser.add_argument('--gpu_avilable', default=False, action='store_true')
    parser.add_argument('--visualization', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--trajectory', default=False, action='store_true')
    args = parser.parse_args()

    if args.method_dir is not None:
        train_parameter_file = os.path.join(args.method_dir, os.path.basename(args.train_parameter))
        if args.imitation_learning:
            model_weights = os.path.join(args.method_dir, 'imitationlearning_model.pth')
        else:
            if os.path.exists(os.path.join(args.method_dir, 'resumed_reinforcementlearning_model.pth')):
                model_weights = os.path.join(args.method_dir, 'resumed_reinforcementlearning_model.pth')
            else:
                model_weights = os.path.join(args.method_dir, 'reinforcementlearning_model.pth')
    else:
        train_parameter_file = args.train_parameter

    # 设置日志和设备
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_avilable else "cpu")
    logging.info('Device: %s', device)

    # 确定测试方案
    method = method_factory[args.method]()
    train_parameter = configparser.RawConfigParser()
    train_parameter.read(train_parameter_file)
    method.configure(train_parameter)
    if method.trainable:
        if args.method_dir is None:
            parser.error('directory error')
        method.get_model().load_state_dict(torch.load(model_weights))

    # 配置gym
    train_parameter = configparser.RawConfigParser()
    train_parameter.read(train_parameter_file)
    env = gym.make('PedestrianRich-v2020')
    env.configure(train_parameter)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(train_parameter, 'robot')
    robot.set_method(method)
    env.set_robot(robot)
    experience = Experience(env, robot, device, gamma=0.9)

    method.set_phase(args.phase)
    method.set_device(device)
    # 设置安全距离
    if isinstance(robot.method, ORCA):
        if robot.visible:
            robot.method.safety_space = 0
        else:
            robot.method.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.method.safety_space)

    method.set_env(env)
    robot.print_info()
    if args.visualization:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            last_pos = current_pos
        if args.trajectory:
            env.render('trajectory', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time-0.25, info)
        if robot.visible and info == 'reach goal':
            pedestrian_times = env.get_pedestrian_times()
            logging.info('Average time  pedestrians arrive goal: %.2f', sum(pedestrian_times) / len(pedestrian_times))
    else:
        experience.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
