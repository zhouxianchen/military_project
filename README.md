1. # military_project
   Project for 一种 复杂战场环境下目标实体类型识别的鲁棒图神经网络方法

   ## Requriements

   Check the requirements.txt

   ## Usage

   handle_data.py describes how to preprocess the military data.

   train_roGCN_dtw.py and train_roGCN_dist.py describes the algorithm.

   ### Example

   ```python
   python train_roGCN_dtw.py --threshold 0.4 --only_gcn --sim_time 3690 --dtw_initial_time 0 --noise_level 0
   ```

   ## Dataset 

   the data are located in logdata2

   ## 

   

   

