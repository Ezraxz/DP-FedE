# DP-FedE

## 参数说明
--data_path: 数据路径
--attack_data_path: 攻击构造数据路径
     
--name: 名称
--state_dir: 保存模型路径 
--log_dir: 日志路径
--tb_log_dir: 
--attack_embed_dir: 攻击时，读取的模型路径
                            
--attack_res_dir: 攻击结果保存路径
    
--model: 算法选择
--batch_size: 训练批次大小
--test_batch_size: 测试批次大小
--num_neg: 一个正样本对应的负样本数轮
--lr: 学习率
    
--model_mode: 如果需要模型融合，选择fusion，然后修改main.py 107行的路径
--is_single: 是否客户端本地训练
    
### for FedE
--num_client: 客户端数
--max_round: 最大轮数
--local_epoch: 本地一轮跑多少epoch
--fraction: 每次聚合选择多少比例的客户端
--log_per_round: 日志记录间隔
--check_per_round: 输出指标间隔

--early_stop_patience: 提前终止耐心
--gamma: fede训练超参
--epsilon: 超参
--hidden_dim: 一个嵌入向量的维度
--gpu: 选择哪个gpu
--num_cpu: 核数
--adversarial_temperature: 
    
### defense_param
--use_dp: 是否dp训练
--naive: 是否简单dp
--microbatch_size: 默认1
--l2_norm_clip: dp参数
--noise_multiplier: dp参数
--sgd_eps: sgd的隐私预算上限
--topk_eps: topk隐私预算
--diff_mrr: 动态调整噪声的阈值
--decline_mult: 动态调整噪声的幅度
    
### attack_param
--is_attack: 是否攻击
--attack_type: 攻击类型
--target_client: 默认0
--attack_client: 默认1
--test_data_count: 攻击测试样本数
--start_round: 开始轮数 默认0
    
### attack-1_param 客户端被动
--threshold_attack1: 攻击1阈值
    
### attack-2_param 客户端主动
--threshold_attack2: 阈值
--cmp_round: 比较轮数 默认1
    
### attack-3_param
--rel_num_multiple: 起始阈值 在攻击中会自动到1.5
    
### test
--test_mode: 默认fake，normal只在制作攻击虚假数据集时使用

## 使用

数据都在data的压缩包，解压一下

以下命令需要针对不同数据集和算法，都要做修改


### 正常Fede
--use_dp: False
--is_attack: False

### DP-Fede
--use_dp: True
--naive: False
--decline_mult : 1 (1 等于不改变噪声)
--is_attack: False

### DP-动态-Fede
--use_dp: True
--naive: False
--decline_mult : 小于1 (1 等于不改变噪声)
--is_attack: False

### DP-简单-Fede
--use_dp: True
--naive: True
--decline_mult : 1 (1 等于不改变噪声)
--is_attack: False

### 单独客户端 DP
--use_dp: True
--naive: True
--decline_mult : 1 (1 等于不改变噪声)
--is_single: True
--is_attack: False

### 单独客户端
--use_dp: False
--is_attack: False
--is_single: True

### 客户端攻击
其他训练参数与对应训练方法一样
--is_attack: True
---attack_type: client

### 服务器攻击
其他训练参数与对应训练方法一样
--is_attack: True
---attack_type: server

### 联合攻击
其他训练参数与对应训练方法一样
--is_attack: True
---attack_type: collusion

攻击介绍：
联合攻击代表一个攻击客户端和服务器联合起来，攻击目标客户端。攻击者同时拥有客户端和服务器的背景知识。本方法设计的联合攻击，攻击方法基于客户端的攻击，但是在获取目标客户端信息时借助服务器获取了更加精准的信息。具体而言，客户端获取的实体嵌入不再是服务器聚合之后的嵌入，而是目标客户端上传给服务器的实体嵌入：

1. 服务器获取目标客户端上传的实体嵌入矩阵之后，直接将其转发给攻击客户端；
2. 攻击者利用服务器转发的目标嵌入，使用基于客户端的攻击方法进行攻击。

相比于攻击者仅仅是客户端的时候，联合攻击使得攻击客户端能够直接获取目标模型的信息，而非经过“稀释”的信息，从而提高攻击成功率。