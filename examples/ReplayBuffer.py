from gymnasium import spaces
import numpy as np
class ReplayBuffer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 初始化只执行一次
        if not hasattr(self, '_initialized'):
            self.S = []  # 状态列表
            self.A = []  # 动作列表
            self.R = []  # 奖励列表
            self.nS = []  # 下一个状态列表
            self.score = 0.0  # 当前回合的总得分
            self._initialized = True
            # print(f"ReplayBuffer initialized: {id(self)}")

        # 定义状态空间和动作空间
        self.observation_space = spaces.Dict({

            "running_tasks": spaces.Discrete(500),  # 假设最多500个运行中的任务
            "waiting_tasks": spaces.Discrete(500),  # 假设最多500个等待中的任务
            "finished_tasks": spaces.Discrete(500),  # 假设最多500个已完成任务，增加了已完成任务的数量

            "ignored_seq_ids": spaces.Discrete(100),  # 被忽略的任务 ID（离散空间） ignored_seq_ids
            "preempted_seq_ids": spaces.Discrete(100), # 被预占的任务 ID（离散空间） preempted_seq_ids

            "max_parallel_tasks": spaces.Discrete(200),  # 假设最大并行任务数为200 scheduler_config.max_num_seqs

            "current_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), #目前的时间 now

            # "dynamic_chunking": spaces.Discrete(2),  # 0或1，是否启用动态分块 self.enable_dynamic_chunking_schedule

            # "low_chunk_size": spaces.Box(low=0, high=2048, shape=(1,), dtype=np.int32), # 低 chunk 大小（连续空间） low_chunk_size
            # "high_chunk_size": spaces.Box(low=0, high=2048, shape=(1,), dtype=np.int32), # 高 chunk 大小（连续空间）high_chunk_size
            # "max_tokens_per_stage": spaces.Box(low=0, high=2048, shape=(1,), dtype=np.int32), # 每阶段最大 token 数（连续空间）chunk_schedule_max_tokens
            # "chunk_schedule_stages": spaces.Discrete(50),  # 假设最多50个阶段， chunk 调度阶段数（离散空间） chunk_schedule_stages

            "policy": spaces.Discrete(3),  # 调度策略（例如 FCFS, 优先级策略等），假设最多3种调度策略
            "can_append_slot": spaces.Discrete(2),  # 0或1，表示是否可以追加解码槽
            "num_free_gpu_blocks": spaces.Discrete(500),  # 当前可用的GPU解码块数，假设最大为500，也就是GPU计算资源使用情况
            "gpu_compute_utilization": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # GPU利用率（连续空间）
            "gpu_memory_utilization": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # 显存使用率（连续空间）


            # 新增任务状态字段，定义为 Box 类型，长度 200，每个任务的状态在 [0, 2] 范围内
            "task_statuses": spaces.Box(low=0, high=2, shape=(500,), dtype=np.int32),  # 任务状态 [0: waiting, 1: running, 2: finished]
            # "tokens_processed": spaces.Box(low=-1, high=2048, shape=(200,), dtype=np.int32),  # 假设最多200个任务，并且每个任务已处理的token数最大值是2048，通过seq.get_num_prompt_tokens_stage_processed()获取
            # "max_tokens": spaces.Box(low=-1, high=2048, shape=(200,), dtype=np.int32), #每个任务的输入序列prompt的token的长度，通过seq.get_prompt_len()获取
            "ttft": spaces.Box(low=-1, high=np.inf, shape=(500,), dtype=np.float32), # 任务的 TTFT 值（连续空间）
            "tbts_avg": spaces.Box(low=-1, high=np.inf, shape=(500,), dtype=np.float32),# 任务的 TBT 平均值（连续空间）
            "tbt_variance": spaces.Box(low=-1, high=np.inf, shape=(500,), dtype=np.float32), # 任务的 TBT 方差（连续空间）
            "tbt_std_dev" : spaces.Box(low=-1, high=np.inf, shape=(500,), dtype=np.float32), # 任务的 TBT 标准差（连续空间）
        })
        
        # 定义动作空间
        # 定义每个参数的离散选择范围
        # 初始化调度参数值
        self.chunk_size_values = [128, 192, 256, 512,1024]
        # self.low_chunk_size_values = [32, 64, 128]
        # self.high_chunk_size_values = [128, 192, 256]
        # self.chunk_schedule_max_tokens_values = [512, 1024, 2048]
        # self.chunk_schedule_stages_values = [2, 4, 6]

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([
            len(self.chunk_size_values)  # chunk_size 选择
            # len(self.low_chunk_size_values),  # low_chunk_size 选择
            # len(self.high_chunk_size_values),  # high_chunk_size 选择
            # len(self.chunk_schedule_max_tokens_values),  # max_tokens_per_stage 选择
            # len(self.chunk_schedule_stages_values),  # chunk_schedule_stages 选择
        ])

    def store(self, state, action, reward, next_state):
        """存储当前的状态、动作、奖励和下一个状态 以及是否终止"""
        self.S.append(state)
        self.A.append(action)
        self.R.append(reward)
        self.nS.append(next_state)


    def accumulate_score(self, reward):
        """累加当前回合的得分"""
        if np.isnan(reward):
            print("Warning: Received a NaN reward. Resetting score to 0.")
            self.score = 0.0
        else:
            self.score += reward
        # print(f"accumulate_score: {self.score}")

    def reset(self):
        """重置所有值"""
        self.S = []  # 状态列表
        self.A = []  # 动作列表
        self.R = []  # 奖励列表
        self.nS = []  # 下一个状态列表
        self.score = 0.0  # 当前回合的总得分
        self.terminated = False  # 当前回合是否终止
    

    def get_data(self):
        
        return self.S, self.A, self.R, self.nS, self.score