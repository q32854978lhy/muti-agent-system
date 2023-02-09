import argparse
import configparser
import logging
import os
import shutil
import sys

import git
import gym
import torch
from coll_gym.envs.methods.method_factory import method_factory
from coll_avo.utils.experience import Experience,ReplayMemory
from coll_avo.utils.trainer import Trainer
from coll_gym.envs.utils.agent import Robot


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--gpu_avilable', default=False, action='store_true')
    parser.add_argument('--train_parameter', type=str, default='parameters/train.config')
    parser.add_argument('--data_dir', type=str, default='data/Attention_rl')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--method', type=str, default='cadrl')
    args = parser.parse_args()
    
    # 配置路径
    make_new_dir = True
    if os.path.exists(args.data_dir):
        key = input('the directory has been created,do you want to overwrite it? (yes/no)')
        if key == 'yes':
            shutil.rmtree(args.data_dir)
        else:
            make_new_dir = False
            args.train_parameter = os.path.join(args.data_dir, os.path.basename(args.train_parameter))
    if make_new_dir:
        os.makedirs(args.data_dir)
        shutil.copy(args.train_parameter, args.data_dir)
    log_file = os.path.join(args.data_dir, 'output.log')
    imitationlearning_weight_file = os.path.join(args.data_dir, 'imitationlearning_model.pth')
    reinforcementlearning_weight_file = os.path.join(args.data_dir, 'reinforcementlearning_model.pth')

    # 配置日志
    file_handler = logging.FileHandler(log_file, mode='a')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_avilable else "cpu")
    logging.info('Device: %s', device)

    # 配置训练方法
    method = method_factory[args.method]()
    if not method.trainable:
        parser.error('Method has to be trainable')
    if args.train_parameter is None:
        parser.error('Method config has to be specified for a trainable network')
    train_parameter = configparser.RawConfigParser()
    train_parameter.read(args.train_parameter)
    method.configure(train_parameter)
    method.set_device(device)

    #配置gym环境
    train_parameter = configparser.RawConfigParser()
    train_parameter.read(args.train_parameter)
    env = gym.make('PedestrianRich-v2020')
    env.configure(train_parameter)
    robot = Robot(train_parameter, 'robot')
    env.set_robot(robot)

    # 获取训练参数
    if args.train_parameter is None:
        parser.error('Train config has to be specified for a trainable network')
    train_parameter = configparser.RawConfigParser()
    train_parameter.read(args.train_parameter)
    reinforcementlearning_learning_rate = train_parameter.getfloat('train', 'reinforcementlearning_learning_rate')
    train_batches = train_parameter.getint('train', 'train_batches')
    train_episodes = train_parameter.getint('train', 'train_episodes')
    sample_episodes = train_parameter.getint('train', 'sample_episodes')
    target_update_interval = train_parameter.getint('train', 'target_update_interval')
    evaluation_interval = train_parameter.getint('train', 'evaluation_interval')
    capacity = train_parameter.getint('train', 'capacity')
    epsilon_start = train_parameter.getfloat('train', 'epsilon_start')
    epsilon_end = train_parameter.getfloat('train', 'epsilon_end')
    epsilon_decay = train_parameter.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_parameter.getint('train', 'checkpoint_interval')

    # 配置经验池
    memory = ReplayMemory(capacity)
    #配置神经网络训练方法（梯度下降）
    model = method.get_model()
    batch_size = train_parameter.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    #配置experience训练数据
    experience = Experience(env, robot, device, memory, method.gamma, target_method=method)

    # 模仿学习初始化神经网络参数
    if os.path.exists(imitationlearning_weight_file):
        model.load_state_dict(torch.load(imitationlearning_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        imitationlearning_episodes = train_parameter.getint('imitation_learning', 'imitationlearning_episodes')
        imitationlearning_method = train_parameter.get('imitation_learning', 'imitationlearning_method')
        imitationlearning_epochs = train_parameter.getint('imitation_learning', 'imitationlearning_epochs')
        imitationlearning_learning_rate = train_parameter.getfloat('imitation_learning', 'imitationlearning_learning_rate')
        trainer.set_learning_rate(imitationlearning_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_parameter.getfloat('imitation_learning', 'safety_space')
        imitationlearning_method = method_factory[imitationlearning_method]()
        imitationlearning_method.multiagent_training = method.multiagent_training
        imitationlearning_method.safety_space = safety_space
        robot.set_method(imitationlearning_method)
        experience.run_k_episodes(imitationlearning_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(imitationlearning_epochs)
        torch.save(model.state_dict(), imitationlearning_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    experience.update_target_model(model)

    # 强化学习设置
    method.set_env(env)
    robot.set_method(method)
    robot.print_info()
    trainer.set_learning_rate(reinforcementlearning_learning_rate)
    # 深度强化学习的具体实现（DRL)
    episode = 0
    while episode < train_episodes:
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        robot.method.set_epsilon(epsilon)

        # 评价模型
        if episode % evaluation_interval == 0:
            experience.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # 经过k次的episodes产生的数据加入的经验池中
        experience.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        #通过数据训练神经网络
        trainer.optimize_batch(train_batches)
        episode += 1
        #更新目标网络
        if episode % target_update_interval == 0:
            experience.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), reinforcementlearning_weight_file)

    # 最终的测试
    experience.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    main()
