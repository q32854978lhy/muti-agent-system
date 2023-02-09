# Attention_RL
提醒:训练的各种参数在coll_avo/parameters/train.config中,以下的教程只是代表训练和测试过程.
一,训练前的准备:
   python3+pip3环境
二,安装第三方库RVO库:
   2.1:安装Cython 
       pip3 install Cython
   2.2:下载并安装RVO
       git clone https://github.com/sybrenstuvel/Python-RVO2.git
       cd Python-RVO2
       python3 setup.py build
       python3 setup.py install
       #此步骤有问题请参考https://github.com/sybrenstuvel/Python-RVO2
       完成之后在python3中测试代码:
       import rvo2 是否成功安装RVO库
三,安装:
   3.1:Attention_rl目录下:
       pip3 install -e .
四,训练(均在coll_avo目录下执行):
   4.1:训练方法指定为attention-rl:
       如果安装了cuda能够使用GPU训练:
       python3 train.py --method attention_rl --data_dir output/attention_rl --gpu_avilable
       如果未安装了cuda仅仅能够使用CPU训练:
       python3 train.py --method attention_rl --data_dir output/attention_rl
   4.2:查看训练结果:
       python3 utils/plotpic.py output/attention_rl/output.log --method Attention-RL
五,测试:
   5.1:执行测试程序:
       python3 test.py --method attention_rl --method_dir output/attention_rl 
       python3 test.py --method orca 
   5.2:查看其中一次测试轨迹:
       python3 test.py --method attention_rl --method_dir output/attention_rl  --visualization --test_case 0 --trajectory
       test_case后接的数字是测试中的一次,可以填写0~测试次数的容量
       python3 test.py --method orca  --visualization --test_case 0 --trajectory
   5.3:动态查看其中一次测试轨迹:
       python3 test.py --method attention_rl --method_dir output/attention_rl  --visualization --test_case 0 
       test_case后接的数字是测试中的一次,可以填写0~测试次数的容量
       python3 test.py --method orca  --visualization --test_case 0 

若需要查看其他方法的表现,只需要将步骤四,五中的所有attention_rl改为cadrl或者lstm_rl.

   
    


     
       
       
   

