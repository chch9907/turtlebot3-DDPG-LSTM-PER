# turtlebot3-DDPG-LSTM-PER
multi-turtlebot3 collision avoidance and navigation via DDPG-LSTM with Prioritized Experience Replay on ROS


## File Structure
* DDPG
  * **model**: DDPG.py
   
  * **environment**: env_ddpg_1.py
   
  * **training**: turtlebot3_ddpg_stage_1
   
  * **launch file**: turtlebot3_ddpg_stage_1.launch
  
  ```bash
  roslaunch turtlebot3_ddpg_stage_1.launch
  ```
 
* DDPG-LSTM-PER:

  * **model**: DDPG_LSTM.py ,  DDPG_PER.py
         
  * **environment**:  env_ddpg_lstm.py
  
  * **training**: turtlebot3_ddpg_lstm_stage_1
  
  * **launch file**: turtlebot3_ddpg_lstm_stage_1.launch
  
  * **utils**: utils.py as supplement for DDPG_PER.py
 
  ```bash
  roslaunch turtlebot3_ddpg_lstm_stage_1.launch
  ```

## Remarks
* Here I just uploaded all the files to repo. You need to put them to the right directories respectively in your environment so as to run the codes successfully.

* It is crucial to modify the launch file since it can change the stages (totally 4 stages in offical package) as well as the nodes to be launched.

* For multi-robot scenario,
  * First you need to modify the stage world file to initial more turtlebots. But I have lost that part of files hence I cannot upload them.
  * Second, you need to dupilcate*env_ddpg_lstm.py* , *turtlebot3_ddpg_lstm_stage_1* and *turtlebot3_ddpg_lstm_stage_1.launch* for each turtlebot with corresponding ids, e.g., '/tb3_1', '/tb3_2', '/tb3_3' for capturing and publishing message respectively. 
  * Third, you can use roslaunch command to launch each turtlebot node to facilitate all the turtlebots to run on one stage simultaneously. It is necessary to consider the interaction between multiple turtlebots on one stage. Particularlly when one of the turtlebot occur a collision, you need to consider whether other turtlebots should respawn since the original respawn process is implemented over the whole stage rather than a turtlebot.  
  
## Blog
[用Turtlebot3实现基于深度强化学习的多移动机器人导航避障的仿真训练](https://blog.csdn.net/Cameron_Rin/article/details/117027106)
