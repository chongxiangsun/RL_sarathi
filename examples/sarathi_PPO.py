import asyncio
import datetime
import logging
import os
import subprocess
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from gymnasium import spaces
from examples.ReplayBuffer import ReplayBuffer
from sarathi.config.config import MetricsConfig, ModelConfig, ParallelConfig, ReplicaConfig, SarathiSchedulerConfig, SystemConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi import  SamplingParams
from examples.RL_manager import RLManager
import tensorflow as tf
import numpy as np
import gym
import copy
import pandas as pd


matplotlib.use('Agg')

import logging


# 在全局范围内初始化 output_dir
BASE_OUTPUT_DIR = "/home/srxh03/sarathi_serve_main/examples/RL_results/RL_offline_inference_output"
output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

def configure_logging(episodes):
    # 动态构造文件名
    log_filename = f"/home/srxh03/sarathi_serve_main/examples/RL_results/log_test_RL_episodes_{episodes}.txt"
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )


# 初始化异步推理引擎
#################################################################################################
def init_inference_engine():
    
    from sarathi.engine.async_llm_engine import AsyncLLMEngine
   
 
    # 配置模型信息
    model_config = ModelConfig(
        model="Qwen/Qwen-7B",  # 模型名称
        dtype="auto",  # 自动数据类型
        seed=42,  # 设置随机种子确保可重复性
        trust_remote_code=True,  # 信任远程代码
        max_model_len=2048  # 例如，将最大长度从32768减少到1024
    )
        
    replica_config = ReplicaConfig(
        output_dir=output_dir,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    scheduler_config = SarathiSchedulerConfig(
        chunk_size=64,
        max_num_seqs=200,#最大批次数
        enable_dynamic_chunking_schedule=False
        # low_chunk_size=32,  # 给定默认值
        # high_chunk_size=128,  # 给定默认值
        # chunk_schedule_max_tokens=1024,  # 给定默认值
        # chunk_schedule_stages=4  # 给定默认值
    )

    metrics_config = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=True,
    )

    system_config = SystemConfig(
        replica_config=replica_config,
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        metrics_config=metrics_config,
    )

    engine = AsyncLLMEngine.from_system_config(system_config)

    return engine

# 定义用于处理推理任务的异步函数
async def handle_request(request_id: int, prompt: str) -> str:
    
     # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  # 温度参数
        top_p=0.9,  # nucleus sampling的top-p参数
        top_k=20,  # top-k采样
        max_tokens=100,  # 最大生成的token数
    )
    output_text = ""
    async for output in engine.generate(
        request_id=str(request_id), prompt=prompt, sampling_params=sampling_params
    ):
        output_text += output.text  # 累积生成的文本内容
    # 仅记录最终输出
    # logging.info(f"Request {request_id} complete output: {output_text}")
    return output_text


# 定义异步推理函数，执行模型的并发推理任务
async def async_inference():
    logging.info("开始异步推理任务...")  # 记录推理开始

    # 记录推理开始的全局时间
    global_start_time = time.time()
    # 记录结果
    results = []
    tasks = []
    all_successful = True  # 用于记录所有任务是否成功完成
    # 遍历每个prompt，创建任务
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Sending requests"):
        prompt = row["prompt"]
        task = asyncio.create_task(handle_request(i, prompt))
        tasks.append(task)

    # 等待所有任务完成并收集结果
    for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Receiving results")):
        try:
            result = await task
            results.append({"request_id": i, "output": result})
        except Exception as e:
            logging.error(f"请求处理失败 (request_id={i}): {e}")
            all_successful = False  # 如果有任务失败，则标记为 False

    # 记录推理结束的全局时间
    global_end_time = time.time()
    logging.info(f"总推理时间: {global_end_time - global_start_time:.2f}秒")

    # 将结果保存为 DataFrame 并写入 CSV 文件
    output_df = pd.DataFrame(results)
    output_df.to_csv(f"{output_dir}/inference_results.csv", index=False)
    logging.info(f"推理结果已保存到 {output_dir}/inference_results.csv")

    return all_successful  # 返回推理是否成功完成

    

#只获得初始状态空间的长度，状态向量化，将复杂状态字典转换为神经网络可以接受的输入量
def preprocess_state(state):
    """
    将复杂状态字典转换为神经网络可以接受的输入向量。
    Args:
        state (dict): 输入状态字典，包含离散和连续特征。
    Returns:
        np.ndarray: 展平后的状态向量。
    """
    state_vector = []
    
    # 1. 离散特征直接添加
    state_vector.append(state['running_tasks'])  # 当前运行任务数
    state_vector.append(state['waiting_tasks'])  # 当前等待任务数
    state_vector.append(state['ignored_seq_ids'])  # 被忽略任务 ID
    state_vector.append(state['preempted_seq_ids'])  # 被预占任务 ID
    state_vector.append(state['max_parallel_tasks'])  # 最大并行任务数
    state_vector.append(state['policy'])  # 当前调度策略
    state_vector.append(state['current_time'].sample()[0] / 1000)  # 当前时间归一化
    # state_vector.append(state['dynamic_chunking'])  # 动态分块开关
    # state_vector.append(state['low_chunk_size'].sample()[0] / 2048)  # 低分块大小归一化
    # state_vector.append(state['high_chunk_size'].sample()[0] / 2048)  # 高分块大小归一化
    # state_vector.append(state['max_tokens_per_stage'].sample()[0] / 2048)  # 每阶段最大 token 数归一化
    task_statuses = state['task_statuses'].sample()

    state_vector.extend(task_statuses.tolist())
    
    # 2. 连续特征展平并归一化
    # tokens_processed = state['tokens_processed']
    # max_tokens = state['max_tokens']

    # tokens_processed = tokens_processed.sample()  # 获取 Box 对象的样本值
    # max_tokens = max_tokens.sample()  # 获取 Box 对象的样本值

    # 用一个较小的常数来填充零值，以避免它们在归一化时的问题
    epsilon = 1e-6
    # tokens_processed = np.where(tokens_processed == 0, epsilon, tokens_processed)  # 将零替换为小常数
    # max_tokens = np.where(max_tokens == 0, epsilon, max_tokens)  # 同理处理

    # state_vector.extend(tokens_processed / 2048)  # 每任务已处理 token 数
    # state_vector.extend(max_tokens / 2048)  # 每任务最大 token 数

    # 处理其他特征（如 ttft, tbts_avg, tbt_variance, tbt_std_dev）
    ttft = state['ttft'].sample()
    tbts_avg = state['tbts_avg'].sample()
    tbt_variance = state['tbt_variance'].sample()
    tbt_std_dev = state['tbt_std_dev'].sample()

    # 对于 ttft、tbts_avg 等，零值处理为 epsilon
    ttft = np.where(ttft == 0, epsilon, ttft)
    tbts_avg = np.where(tbts_avg == 0, epsilon, tbts_avg)
    tbt_variance = np.where(tbt_variance == 0, epsilon, tbt_variance)
    tbt_std_dev = np.where(tbt_std_dev == 0, epsilon, tbt_std_dev)

    state_vector.extend(ttft / 2048)  # TTFT 归一化
    state_vector.extend(tbts_avg / 2048)  # TBT 平均值归一化
    state_vector.extend(tbt_variance / 2048)  # TBT 方差归一化
    state_vector.extend(tbt_std_dev / 2048)  # TBT 标准差归一化
    
    # print("State vector:", state_vector)
    
    return np.array(state_vector)

# 熵动态调整
def get_entropy_weight(episode, max_episodes, base_weight=0.01, temperature=0.1):
    """
    根据温度系数动态调整熵权重
    :param episode: 当前训练轮次
    :param max_episodes: 总训练轮次
    :param base_weight: 熵奖励的基准权重
    :param temperature: 温度系数，控制熵权重衰减速度
    :return: 当前熵权重
    """
    return base_weight * np.exp(-episode / (max_episodes * temperature))


def read_config():
    # 读取当前的训练回合数和运行次数
    try:
        with open("config.txt", "r") as f:
            lines = f.readlines()
            # 通过分隔符 '=' 提取数字部分
            episodes = int(lines[0].strip().split('=')[1])  # 当前训练的回合数
            run_count = int(lines[1].strip().split('=')[1])  # 当前运行的次数
    except FileNotFoundError:
        # 如果配置文件不存在，默认值
        episodes = 50
        run_count = 0
    return episodes, run_count


# 更新配置文件函数
def update_config(episodes, run_count):
    with open("config.txt", "w") as f:
        f.write(f"episodes={episodes}\n")  # 以 key=value 格式保存
        f.write(f"run_count={run_count}\n")  # 以 key=value 格式保存


# 最大运行次数限制
max_run_count = 5 # 设置运行最大次数

# 主程序
if __name__ == '__main__':
    # 读取当前的训练回合数和脚本运行次数
    episodes, run_count = read_config()

    # 配置日志
    configure_logging(episodes)

    # 获取 ReplayBuffer 对象
    buffer = ReplayBuffer()

    # 获取状态空间和动作空间的维度
    state_dim = len(preprocess_state(buffer.observation_space))  
    action_dim = len(buffer.action_space.nvec)

    A_learning_rate = 3e-4  # Actor 网络的学习率
    C_learning_rate = 3e-4  # Critic 网络的学习率

    # 初始化 Actor 网络和 Critic 网络
    RLManager.initialize(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=A_learning_rate,
        critic_lr=C_learning_rate,
        action_dim_list=[5]  # 每个动作维度的取值范围, 根据ReplayBuffer中定义的动作空间决定
    )

    actor = RLManager.get_actor()
    critic = RLManager.get_critic()

    # 设置最大训练回合数
    gamma = 0.9  # 折扣因子
    lam = 0.98  # Generalized Advantage Estimation (GAE) 的 λ
    K_epoch = 10  # 每个回合的训练轮数

    plot_score = []  # 用于记录每个回合的得分

    # 从CSV文件中读取数据集，包含一列prompt，限制读取前500行
    df = pd.read_csv("/home/srxh03/sarathi_serve_main/examples/datasets/test_10000.csv", header=None, names=["prompt"], nrows=500)

    engine = init_inference_engine()  # 初始化推理引擎
    
    for e in range(episodes):  # 进行训练的每一个回合    
        done = asyncio.run(async_inference()) 

        S, A, R, nS = [], [], [], []  # 初始化记录状态、动作、奖励和下一个状态的列表
        score = 0.0  # 当前回合的总得分
        if done:
            S, A, R, nS, score = buffer.get_data()
            discounted_r = []
            tmp_r = 0.0
            v_nS = critic.get_value(np.array(nS, dtype=np.float64))
            v_nS[-1] = 0  # 将最后一个状态的价值设为 0

            for r, vs in zip(R[::-1], v_nS[::-1]):
                r = np.nan_to_num(r, nan=0.0)  # 如果r为NaN，则设置为0
                tmp_r = r + gamma * (lam * tmp_r + (1 - lam) * vs[0])
                discounted_r.append(np.array([tmp_r]))
            discounted_r.reverse()

            bs = np.array(S, dtype=np.float64)
            ba = np.array(A)
            br = np.array(discounted_r, dtype=np.float64)

            advantage = critic.get_advantage(bs, br)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            entropy_weight = get_entropy_weight(e, episodes, base_weight=0.05, temperature=0.1)

            for k in range(K_epoch):
                actor.learn(bs, ba, advantage, entropy_weight=entropy_weight)  # 更新 Actor 网络
                critic.learn(bs, br)  # 更新 Critic 网络
            actor.update_policy()
            print(f"Episode: {e + 1}/{episodes}, Score: {score}")
           
            # 重置ReplayBuffer
            buffer.reset()

            plot_score.append(score)

    # 绘制结果并保存图像
    plt.plot(plot_score)

    # 使用 f-string 动态插入 episodes 变量
    plt.savefig(f"task:500_interval:10s_episodes:{episodes}.png")

    # 增加运行次数并更新配置
    run_count += 1

    # 如果运行次数小于最大限制，则更新训练的回合数并重新运行
    if run_count < max_run_count:
        episodes += 50  # 增加训练回合数
        update_config(episodes, run_count)
        print(f"Training run {run_count}/{max_run_count} complete. Starting next run with {episodes} episodes.")
        os.system(f"python {__file__}")  # 重新执行当前脚本
    else:
        print("Max run count reached. Exiting.")