from abc import ABC, abstractmethod
import subprocess
import time
from typing import List
import numpy as np


from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes import scheduler_output
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore

import logging

logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        # we maintain this just for logging purposes
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[Sequence] = []
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []

        self.finnished: List[Sequence] = []
        self.schedule_counter = 0
        self.store_call_counter = 0  # 用于跟踪存储数据的次数

    def reset_state(self) -> None:
        self._iteration_id = -1



    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:

        return self.waiting or self.running

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running)

    @abstractmethod
    # def _schedule(self) -> SchedulerOutputs:
    def _schedule(self) -> tuple[SchedulerOutputs]:
        pass

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.   

        self._iteration_id += 1

        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )
        # 解包 _schedule 的返回值
        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs
    
    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                # 计算 TTFT、TBT、TBT 的方差和标准差
                ttft = seq.calculate_ttft()
                tbts = seq.calculate_tbt()
                tbts_avg = seq.calculate_tbt_avg()
                variance, std_dev = seq.calculate_tbt_variance_and_std()
                
                # 打印总结信息
                logging.info("=== Summary of Requests ===")
                logging.info(f"Request ID: {seq.seq_id}")
                logging.info(f"  TTFT: {ttft:.4f} seconds")
                logging.info(f"  TBTs: {tbts}")
                logging.info(f"  TBTs Avg: {tbts_avg:.4f} seconds")
                logging.info(f"  TBT Variance: {variance:.4f}")
                logging.info(f"  TBT Standard Deviation: {std_dev:.4f}")
                
                # 清理完成的序列
                self._free_seq(seq)
                # 将已完成的任务加进已完成任务列表
                self.finnished.append(seq)
                # print(self.finnished)
        self.running = [seq for seq in self.running if not seq.is_finished()]

# 获取状态空间
    def get_state(self):
        
        # 计算所有任务的 TTFT、TBT、TBT的方差和标准差
        all_sequences = self.finnished + self.running + self.waiting  # 按照顺序：已完成 -> 正在运行 -> 等待

        state = {
        "running_tasks": len(self.running),
        "waiting_tasks": len(self.waiting),
        "finnished_tasks": len(self.finnished),
        "ignored_seq_ids": len(self.ignored_seq_ids),
        "preempted_seq_ids": len(self.preempted_seq_ids),
        "max_parallel_tasks": self.scheduler_config.max_num_seqs,
        "current_time": time.monotonic(),
        "dynamic_chunking": self.enable_dynamic_chunking_schedule,
        "low_chunk_size": self.low_chunk_size,
        "high_chunk_size": self.high_chunk_size,
        "max_tokens_per_stage": self.chunk_schedule_max_tokens,
        "chunk_schedule_stages": self.chunk_schedule_stages,
        "policy": self.policy.__class__.__name__,
        "can_append_slot": self.block_manager.can_append_slot(),
        "num_free_gpu_blocks": self.block_manager.gpu_allocator.get_num_free_blocks(), 
        "gpu_compute_utilization": self.get_gpu_compute_utilization(),
        "gpu_memory_utilization": self.get_gpu_memory_utilization(),


        "task_statuses" : [2 if seq in self.finnished else 1 if seq in self.running else 0 for seq in all_sequences],
        # "tokens_processed": [seq.get_num_prompt_tokens_stage_processed() for seq in all_sequences],
        # "max_tokens": [seq.get_prompt_len() for seq in all_sequences],
        "ttft": [seq.calculate_ttft() for seq in all_sequences],
        "tbts_avg" : [seq.calculate_tbt_avg() for seq in all_sequences],
        "tbt_variance": [seq.calculate_tbt_variance_and_std()[0] for seq in all_sequences],
        "tbt_std_dev": [seq.calculate_tbt_variance_and_std()[1] for seq in all_sequences],
        }

        
        return state
# 计算GPU利用率
    def get_gpu_compute_utilization(self):
        """
        使用 nvidia-smi 命令来获取 GPU 计算资源利用率。
        """
        try:
            # 调用 nvidia-smi 获取 GPU 计算利用率
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            # 解析输出，获取 GPU 计算利用率
            gpu_utilization = int(result.stdout.strip())  # GPU 计算利用率百分比

            # 将计算利用率转换为 0 到 1 之间的浮动值
            gpu_compute_utilization = gpu_utilization / 100.0
            return gpu_compute_utilization

        except subprocess.CalledProcessError as e:
            logger.error("Failed to get GPU compute utilization: %s", e)
            return 0.0  # 返回 0 表示无法获取计算利用率
        except Exception as e:
            logger.error("Error in get_gpu_compute_utilization: %s", e)
            return 0.0  # 出现异常时返回 0
# 计算显存
    def get_gpu_memory_utilization(self):
        """
        使用 nvidia-smi 命令来获取 GPU 显存利用率。
        """
        try:
            # 调用 nvidia-smi 获取显存使用情况，并以 CSV 格式返回
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            # 解析输出，获取显存使用量和总显存量
            memory_used, memory_total = map(int, result.stdout.strip().split(','))
            
            # 计算显存使用率
            gpu_memory_utilization = memory_used / memory_total
            return gpu_memory_utilization

        except subprocess.CalledProcessError as e:
            logger.error("Failed to get GPU memory utilization: %s", e)
            return 0.0  # 返回 0 表示无法获取显存使用率
        except Exception as e:
            logger.error("Error in get_gpu_memory_utilization: %s", e)
            return 0.0  # 出现异常时返回 0

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches -= 1

         # 检查是否所有任务都已完成
        if not self.has_unfinished_seqs():  # 如果没有等待和正在运行的任务
        # 如果所有任务都已完成，清空已完成队列
            # print("All tasks are finished. Clearing the 'finnished' queue.")
            # print("清除计数器")
            self.schedule_counter = 0
            self.finnished.clear()
            self.store_call_counter = 0

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)
        self.waiting.insert(0, seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.pop(0)
            return False

        return True
