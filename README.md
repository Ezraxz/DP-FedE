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
