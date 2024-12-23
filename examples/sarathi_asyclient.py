import asyncio
import datetime
import time
import pandas as pd
import os
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig, SystemConfig, ReplicaConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.engine.async_llm_engine import AsyncLLMEngine

import logging

from sarathi.config.config import VllmSchedulerConfig
logging.basicConfig(
    filename="/home/srxh03/sarathi-serve-main/examples/results/log_test_chunk_new.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)


# 定义异步推理函数，执行模型的并发推理任务
async def async_inference1():
    logging.info("开始异步推理任务...")  # 记录推理开始

    # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  # 温度参数
        top_p=0.9,  # nucleus sampling的top-p参数
        top_k=20,  # top-k采样
        max_tokens=100,  # 最大生成的token数
    )

    BASE_OUTPUT_DIR = "/home/srxh03/sarathi_serve_main/examples/results/offline_inference_output"

    output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

     # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

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

    # scheduler_config = SarathiSchedulerConfig(
    #     chunk_size=64,
    #     max_num_seqs=200,
    #     enable_dynamic_chunking_schedule=True,
    #     low_chunk_size=32,  # 给定默认值
    #     high_chunk_size=128,  # 给定默认值
    #     chunk_schedule_max_tokens=1024,  # 给定默认值
    #     chunk_schedule_stages=4  # 给定默认值
    # )


    scheduler_config = VllmSchedulerConfig(
        max_num_seqs= 200,
        max_batched_tokens=None
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

    # 从CSV文件中读取数据集，包含一列prompt，限制读取前500行
    df = pd.read_csv("/home/srxh03/sarathi-serve-main/examples/datasets/test_10000.csv", header=None, names=["prompt"], nrows=500)

    # 记录推理开始的全局时间
    global_start_time = time.time()

    # 定义用于处理推理任务的异步函数
    async def handle_request(request_id: int, prompt: str) -> str:
        output_text = ""
        async for output in engine.generate(
            request_id=str(request_id), prompt=prompt, sampling_params=sampling_params
        ):
            output_text += output.text  # 累积生成的文本内容
        # 仅记录最终输出
        logging.info(f"Request {request_id} complete output: {output_text}")
        return output_text


    # 记录结果
    results = []
    tasks = []

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

    # 记录推理结束的全局时间
    global_end_time = time.time()
    logging.info(f"总推理时间: {global_end_time - global_start_time:.2f}秒")

    # 保存结果为 DataFrame 并写入 CSV 文件
    output_df = pd.DataFrame(results)
    output_df.to_csv(f"{output_dir}/inference_results.csv", index=False)
    logging.info(f"推理结果已保存到 {output_dir}/inference_results.csv")

# 运行异步推理任务
asyncio.run(async_inference1())


