import copy
import logging
import time
from gymnasium import spaces
from examples.RL_manager import RLManager
import tensorflow as tf
from typing import List
from examples.ReplayBuffer import ReplayBuffer
import numpy as np
from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger


logger = init_logger(__name__)


class SarathiScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.ignored_seq_ids = []  # 作为实例属性
        self.preempted_seq_ids = []  # 作为实例属性
        self.num_batched_tokens = 0  # 添加实例属性，用于跟踪已分配的token总数

        self.chunk_size = self.scheduler_config.chunk_size
        self.enable_dynamic_chunking_schedule = (
            self.scheduler_config.enable_dynamic_chunking_schedule
        )
        # next four params apply only when using dynamic schedule
        self.low_chunk_size = self.scheduler_config.low_chunk_size
        self.high_chunk_size = self.scheduler_config.high_chunk_size
        self.chunk_schedule_max_tokens = self.scheduler_config.chunk_schedule_max_tokens
        self.chunk_schedule_stages = self.scheduler_config.chunk_schedule_stages

        if self.enable_dynamic_chunking_schedule:
            assert self.chunk_schedule_stages > 0
            assert self.chunk_schedule_max_tokens > 0
            assert self.low_chunk_size % 32 == 0
            assert self.high_chunk_size % 32 == 0
            self._chunk_sizes = self._compute_chunk_size_schedule()
            self._tokens_per_stage = int(
                np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages)
            )

    def _compute_chunk_size_schedule(self):
        # create num_steps equally spaced chunk sizes between low_chunk_size and high_chunk_size
        chunk_sizes = np.linspace(
            self.low_chunk_size,
            self.high_chunk_size,
            self.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1]
        # align each chunk size to the nearest multiple of 32 or self.low_chunk_size
        round_of_chunk_sizes = min(32, self.low_chunk_size)
        chunk_sizes = (
            np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        )
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        return chunk_sizes

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        if self.enable_dynamic_chunking_schedule:
            request_stage_idx = int(
                np.ceil(
                    seq.get_num_prompt_tokens_stage_processed()
                    // self._tokens_per_stage
                )
            )
            assert request_stage_idx < len(self._chunk_sizes)
            chunk_size = self._chunk_sizes[request_stage_idx]
        else:
            chunk_size = self.chunk_size

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    #更新调度器参数
    def update_SarathiSchedulerConfig(self, chunk_size_idx):
        """
        根据 RL agent 选择的动作更新调度配置
        :param action: RL agent 选择的动作 (五个整数的向量)
        """

        # 根据索引选择相应的调度参数值
        self.chunk_size = chunk_size_idx
        # self.low_chunk_size = low_chunk_size_idx
        # self.high_chunk_size = high_chunk_size_idx
        # self.chunk_schedule_max_tokens = max_tokens_idx
        # self.chunk_schedule_stages = stages_idx

        # 如果启用了动态调度，重新计算 chunk sizes 和 tokens_per_stage
        # if self.enable_dynamic_chunking_schedule:
        #     self._chunk_sizes = self._compute_chunk_size_schedule()
        #     self._tokens_per_stage = int(np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages))


    #计算奖励函数
    def calculate_reward(self, now, observation):
        """
        根据 TTFT、TBT 平均值、方差和标准差等计算奖励
        :param observation: 当前状态空间字典（observation）
        """
        # 从 observation 中获取指标
        ttft = np.array(observation["ttft"])
        tbts_avg = np.array(observation["tbts_avg"])
        tbt_variance = np.array(observation["tbt_variance"])
        tbt_std_dev = np.array(observation["tbt_std_dev"])
        # current_time = observation["current_time"]  # 当前时间

        # 筛选有效数据，避免 0 的干扰
        valid_ttft = ttft[ttft > 0]
        valid_tbts_avg = tbts_avg[tbts_avg > 0]
        valid_tbt_variance = tbt_variance[tbt_variance > 0]
        valid_tbt_std_dev = tbt_std_dev[tbt_std_dev > 0]

        # 计算 TTFT 的增量
        ttft_increment = valid_ttft[1:] - valid_ttft[:-1]
        
            
        # 计算 TBT 平均值的增量（即每次平均值的差值）
        tbts_avg_increment = valid_tbts_avg[1:] - valid_tbts_avg[:-1]

        # #归一化处理
        # def normalize_data(data):
        #     min_value = np.min(data)
        #     max_value = np.max(data)
        #     return (data - min_value) / (max_value - min_value)

        
        # # 对TTFT 增量进行归一化
        # ttft_increment_normalized = (ttft_increment - ttft_increment.min()) / (ttft_increment.max() - ttft_increment.min() + 1e-6)
        #  # 对 TBT 平均值差值进行归一化
        # tbts_avg_increment_normalized = normalize_data(tbts_avg_increment)

        # # 对每个有效数据进行归一化
        # # ttft_normalized = normalize_data(valid_ttft)
        # # tbts_avg_normalized = normalize_data(valid_tbts_avg)
        # tbt_variance_normalized = normalize_data(valid_tbt_variance)
        # tbt_std_dev_normalized = normalize_data(valid_tbt_std_dev)

        # 归一化处理
        def normalize_data_with_sign(data):
            sign = np.sign(data)
            abs_data = np.abs(data)
            normalized = (abs_data - np.min(abs_data)) / (np.max(abs_data) - np.min(abs_data) + 1e-6)
            return normalized * sign  # 保留符号

        # 对 TTFT 增量进行归一化（包括符号）
        ttft_increment_normalized = normalize_data_with_sign(ttft_increment)
        
        # 对 TBT 平均值差值进行归一化（包括符号）
        tbts_avg_increment_normalized = normalize_data_with_sign(tbts_avg_increment)

        # 对 TBT 方差和标准差进行归一化
        tbt_variance_normalized = normalize_data_with_sign(valid_tbt_variance)
        tbt_std_dev_normalized = normalize_data_with_sign(valid_tbt_std_dev)

        # 计算每个量的分数
        score_ttft = ttft_increment_normalized - 1
        score_tbts_avg = tbts_avg_increment_normalized - 1
        score_tbt_variance = tbt_variance_normalized - 1
        score_tbt_std_dev = tbt_std_dev_normalized - 1
        
        # 奖励权重参数
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0


        # 奖励计算
        reward = (
            alpha * score_ttft.mean()  # 惩罚较大的 TTFT
            + beta * score_tbts_avg.mean()  # 惩罚较大的 TBT 平均值
            + gamma * score_tbt_variance.mean()  # 惩罚较大的 TBT 方差
            + delta * score_tbt_std_dev.mean()  # 惩罚较大的 TBT 标准差
        )

        # Reward Rescaling：限制奖励值范围或进行缩放
        reward = reward * 0.1  # 奖励乘以系数 0.5（小于 1）

        return reward

    #状态向量化，将复杂状态字典转换为神经网络可以接受的输入量
    def preprocess_state(self, state):
        """
        将复杂状态字典转换为神经网络可以接受的输入向量。
        
        Args:
            state (dict): 输入状态字典，包含离散和连续特征。
            
        Returns:
            np.ndarray: 展平后的状态向量。
        """
        # 查看输入的状态
        # print("State dictionary:", state)

        state_vector = []
        
        # 1. 离散特征直接添加
        state_vector.append(state['running_tasks'])  # 当前运行任务数
        state_vector.append(state['waiting_tasks'])  # 当前等待任务数
        state_vector.append(state['ignored_seq_ids'])  # 被忽略任务 ID
        state_vector.append(state['preempted_seq_ids'])  # 被预占任务 ID
        state_vector.append(state['max_parallel_tasks'])  # 最大并行任务数
        
        
        # 对于 'policy'，如果是字符串，进行处理（如映射为数值）
        policy = state['policy']
        if isinstance(policy, str):  # 假设 policy 可能是 'FCFS', 'SJF', etc.
            policy_mapping = {'FCFS': 1, 'SJF': 2, 'RoundRobin': 3}  # 示例映射
            policy = policy_mapping.get(policy, -1)  # 默认 -1 作为未知策略
        state_vector.append(policy)

        # 注意，这里需要确保 `state['current_time']` 和分块大小的处理
        current_time = state['current_time']
        # state_vector.append(state['dynamic_chunking'])  # 动态分块开关
        # low_chunk_size = state['low_chunk_size']
        # high_chunk_size = state['high_chunk_size']
        # max_tokens_per_stage = state['max_tokens_per_stage']

        # 当前时间归一化
        state_vector.append(current_time / 1000)  # 假设当前时间单位为毫秒，归一化到秒

        # 分块大小归一化
        # state_vector.append(low_chunk_size / 2048)  # 低分块大小归一化
        # state_vector.append(high_chunk_size / 2048)  # 高分块大小归一化
        # state_vector.append(max_tokens_per_stage / 2048)  # 每阶段最大 token 数归一化

        task_statuses = state['task_statuses']

        state_vector.extend(task_statuses)
        # 2. 连续特征展平并归一化
        # tokens_processed 和 max_tokens
        # tokens_processed = state['tokens_processed']
        # max_tokens = state['max_tokens']

        #  # 确保tokens_processed和max_tokens不为空，填充为200长度
        # if len(tokens_processed) < 200:
        #     tokens_processed = tokens_processed + [0] * (200 - len(tokens_processed))  # 填充至200长度
        # if len(max_tokens) < 200:
        #     max_tokens = max_tokens + [0] * (200 - len(max_tokens))  # 填充至200长度

        # 这里不再使用 `.sample()`，直接使用列表的元素
        # 确保 tokens_processed 和 max_tokens 的元素是数值类型
        # tokens_processed = np.array(tokens_processed)
        # max_tokens = np.array(max_tokens)

        

        # 用一个较小的常数来填充零值，以避免它们在归一化时的问题
        epsilon = 1e-6
        # tokens_processed = np.where(tokens_processed == 0, epsilon, tokens_processed)  # 将零替换为小常数
        # max_tokens = np.where(max_tokens == 0, epsilon, max_tokens)  # 同理处理

        # 将每个任务的 token 数量归一化到 0-1 范围内
        # state_vector.extend(tokens_processed / 2048)  # 每任务已处理 token 数
        # state_vector.extend(max_tokens / 2048)  # 每任务最大 token 数

        # 处理其他特征（如 ttft, tbts_avg, tbt_variance, tbt_std_dev）
        ttft = state['ttft']
        tbts_avg = state['tbts_avg']
        tbt_variance = state['tbt_variance']
        tbt_std_dev = state['tbt_std_dev']

        # 确保这些特征不为空，填充为200长度
        if len(ttft) < 500:
            ttft = ttft + [0] * (500 - len(ttft))  # 填充至200长度
        if len(tbts_avg) < 500:
            tbts_avg = tbts_avg + [0] * (500 - len(tbts_avg))  # 填充至200长度
        if len(tbt_variance) < 500:
            tbt_variance = tbt_variance + [0] * (500 - len(tbt_variance))  # 填充至200长度
        if len(tbt_std_dev) < 500:
            tbt_std_dev = tbt_std_dev + [0] * (500 - len(tbt_std_dev))  # 填充至200长度

        # 同样对 ttft, tbts_avg 等进行处理，避免零值
        ttft = np.array(ttft)
        tbts_avg = np.array(tbts_avg)
        tbt_variance = np.array(tbt_variance)
        tbt_std_dev = np.array(tbt_std_dev)

        # 对于 ttft、tbts_avg 等，零值处理为 epsilon
        ttft = np.where(ttft == 0, epsilon, ttft)
        tbts_avg = np.where(tbts_avg == 0, epsilon, tbts_avg)
        tbt_variance = np.where(tbt_variance == 0, epsilon, tbt_variance)
        tbt_std_dev = np.where(tbt_std_dev == 0, epsilon, tbt_std_dev)

        # 将这些特征归一化
        state_vector.extend(ttft / 2048)  # TTFT 归一化
        state_vector.extend(tbts_avg / 2048)  # TBT 平均值归一化
        state_vector.extend(tbt_variance / 2048)  # TBT 方差归一化
        state_vector.extend(tbt_std_dev / 2048)  # TBT 标准差归一化

        

        # # 打印出处理后的状态向量
        # print("State vector:", state_vector)
        # print(len(state_vector))
        # 返回展平后的状态向量（numpy 数组）
        return np.array(state_vector, dtype=np.float32)



    def _schedule(self) -> tuple[SchedulerOutputs, dict, float, bool]:
        self.schedule_counter += 1  # 更新计数器

        if self.schedule_counter >= 10:
            
            # 获取 ReplayBuffer 实例
            buffer = ReplayBuffer()

            # Step 1: 获取当前状态
            current_state = self.get_state()
            flattened_state = self.preprocess_state(current_state)# 展平状态空间
            flattened_state = copy.deepcopy(flattened_state)  # 更新状态，使用深拷贝以防止状态被意外修改

            # Step 2: RL agent 根据当前状态选择动作
            actor = RLManager.get_actor()
            assert actor is not None, "Actor has not been initialized. Call initialize_actor_critic() first."

            
            action, policy = actor.choice_action(flattened_state)

            
            # 定义局部变量
            chunk_size_values = [128, 192, 256, 512,1024]

            # 新的动作
            # 更新调度器配置
            self.update_SarathiSchedulerConfig(
                chunk_size_values[action]
            )

        # 继续原有逻辑
        now = time.monotonic()

        running: List[Sequence] = []
        
        self.ignored_seq_ids.clear()  # 清空之前的记录
        self.preempted_seq_ids.clear()  # 清空之前的记录
        ignored_seq_ids = self.ignored_seq_ids
        preempted_seq_ids = self.preempted_seq_ids

        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        # 使用实例属性 num_batched_tokens
        self.num_batched_tokens = 0  # 初始化 num_batched_tokens


        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                self.num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )


        for seq in running_prefills:
            assert not seq.prompt_stage_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, self.num_batched_tokens
            )


            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            self.num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        while self.waiting:
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, self.num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            self.num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running


        if self.schedule_counter >= 10:
            self.schedule_counter = 0  # 每30次调用后重置计数器
            # 获取调度后的状态空间
            observation = self.get_state()
            flattened_next_state = self.preprocess_state(observation)


            # 计算奖励函数,稍后处理
            reward = self.calculate_reward(now,observation)

            # print(f"Reward: {reward}")

            buffer.store(flattened_state, action, reward, flattened_next_state)
            buffer.accumulate_score(reward)
                # self.store_call_counter += 1
            

        scheduler_outputs = SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )


        # 返回原有结果、新的状态空间和奖励值
        return scheduler_outputs
        
