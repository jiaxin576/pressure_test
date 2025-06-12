#coding=utf-8
import threading,os
import requests
import datetime
import time
import json
import logging
import argparse
import requests
import json
import datetime
from Crypto.Cipher import DES
import base64
import requests
import json
import binascii 
from Crypto.Util.Padding import pad, unpad
import time
import random
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import concurrent.futures
import numpy as np  # 引入numpy用于计算百分位耗时，更精确
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import platform

SEVER_URL = "https://voice-test.ztems.com/zte/nlu/api/v1/ask"
HTTP_SOCKET_TIME_OUT = 15000


# --- 新接口的配置 ---
# 默认配置，可以通过命令行参数覆盖
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# !! 重要: 请将此处的默认Token和Model替换为您的真实值，或通过命令行参数传入 !!
DEFAULT_API_TOKEN = "" 
DEFAULT_MODEL_NAME = "" # 例如 Doubao-pro-32k
HTTP_TIMEOUT = 60  # 秒

# --- 主日志记录器 (记录测试流程和统计信息) ---
log_formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 主日志文件处理器
main_log_handler = logging.FileHandler("pressure_test.log", mode='w', encoding='utf-8')
main_log_handler.setFormatter(log_formatter)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# 获取并配置根logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(main_log_handler)
logger.addHandler(console_handler)

# --- API请求/响应日志记录器 (只记录完整的API交互) ---
api_io_logger = logging.getLogger('api_io')
api_io_logger.setLevel(logging.INFO)
api_io_handler = logging.FileHandler("api_requests.log", mode='w', encoding='utf-8')
api_io_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s\n%(message)s\n'))
api_io_logger.addHandler(api_io_handler)
api_io_logger.propagate = False # 防止将日志消息传递给根logger
# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（Windows/Linux通用，macOS可用['Heiti SC']）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题
# 过滤掉第三方库的低级别日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ==============================================================================
# 2. API 接口抽象 (新功能，方便替换)
# ==============================================================================

class BaseAPI(ABC):
    """API处理器的抽象基类，定义了所有API实现必须遵循的接口。"""
    def __init__(self, api_url: str):
        self.api_url = api_url

    @abstractmethod
    def prepare_request(self, prompt: str):
        """
        准备API请求。
        :param prompt: 用户输入的提示词
        :return: 一个元组，包含 (headers, payload)
        """
        pass

class VolcAPI(BaseAPI):
    def __init__(self, api_url: str, api_token: str, model_name: str):
        super().__init__(api_url)
        self.api_token = api_token
        self.model_name = model_name

    def prepare_request(self, prompt: str) -> tuple[dict, dict]:
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是人工智能助手，名为豆包。"},
                {"role": "user", "content": prompt}
            ]
        }
        return headers, payload

# ==============================================================================
# 3. 核心测试逻辑
# ==============================================================================

def get_random_message():
    """随机选择一个消息，使测试更真实"""
    messages = [
        "你好，请介绍一下你自己",
        "请帮我写一首关于春天的五言绝句",
        "解释一下什么是“人工智能”",
        "北京今天的天气怎么样？",
        "给我讲一个关于程序员的笑话",
        "如何用Python实现一个快速排序算法？"
    ]
    return random.choice(messages)

def send_request(api_handler: BaseAPI, prompt: str):
    """
    发送单个HTTP请求的函数。
    :param api_handler: 一个实现了BaseAPI的实例
    :param prompt: 请求的prompt
    :return: 一个包含结果的元组 (status, duration, error_msg)
    """
    headers, payload = api_handler.prepare_request(prompt)
    
    # 【日志增强】记录完整的请求体
    api_io_logger.info(f"--- REQUEST -->\n{json.dumps(payload, indent=2, ensure_ascii=False)}")

    start_time = time.monotonic()
    try:
        response = requests.post(
            url=api_handler.api_url,
            headers=headers,
            json=payload,
            timeout=HTTP_TIMEOUT
        )
        duration = time.monotonic() - start_time
        
        # 【日志增强】记录完整的响应体
        api_io_logger.info(f"--- RESPONSE <-- (Status: {response.status_code}, Duration: {duration:.2f}s)\n{response.text}\n" + "="*80)

        if response.status_code == 200:
            try:
                response.json() # 验证JSON是否有效
                logger.info(f'请求成功 - 耗时: {duration:.2f}s - 输入: "{prompt[:20]}..."')
                return ("success", duration, None)
            except json.JSONDecodeError:
                logger.error(f'请求成功但响应非JSON - 耗时: {duration:.2f}s - 响应: {response.text[:200]}')
                return ("failed", duration, "Invalid JSON response")
        else:
            logger.error(f'请求失败 - 状态码: {response.status_code} - 耗时: {duration:.2f}s - 响应: {response.text[:200]}')
            return ("failed", duration, f"HTTP {response.status_code}")
            
    except requests.RequestException as e:
        duration = time.monotonic() - start_time
        error_msg = str(e)
        logger.error(f'请求异常 - 耗时: {duration:.2f}s - 异常: {e}')
        # 【日志增强】在发生异常时也记录日志
        api_io_logger.error(f"--- EXCEPTION <-- (Duration: {duration:.2f}s)\n{error_msg}\n" + "="*80)
        return ("exception", duration, error_msg)

def run_test_stage(concurrency: int, duration_minutes: float, api_handler: BaseAPI):
    """
    运行一个测试阶段。
    :return: 包含该阶段统计结果的字典。
    """
    logger.info("="*80)
    logger.info(f"开始测试阶段: 并发数 = {concurrency}, 持续时间 = {duration_minutes} 分钟, 模型 = {getattr(api_handler, 'model_name', 'N/A')}")
    logger.info("="*80)

    duration_seconds = duration_minutes * 60
    end_time = time.monotonic() + duration_seconds

    latencies, success_count, failed_count, exception_count = [], 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f'Worker-C{concurrency}') as executor:
        futures = {executor.submit(send_request, api_handler, get_random_message()) for _ in range(concurrency)}
        
        while futures:
            # 等待下一个任务完成
            done_future = next(concurrent.futures.as_completed(futures))
            futures.remove(done_future)

            # 处理已完成任务的结果
            status, resp_time, _ = done_future.result()
            if status == "success":
                success_count += 1
                latencies.append(resp_time) # 只统计成功请求的延迟
            elif status == "failed":
                failed_count += 1
            else: # exception
                exception_count += 1

            # 如果测试时间未结束，提交新任务
            if time.monotonic() < end_time:
                new_future = executor.submit(send_request, api_handler, get_random_message())
                futures.add(new_future)

    # --- 阶段结束，计算并返回统计报告 ---
    total_requests = success_count + failed_count + exception_count
    stage_results = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "success_count": success_count,
        "failed_count": failed_count,
        "exception_count": exception_count,
        "rps": 0,
        "success_rate": 0,
        "avg_time": 0,
        "min_time": 0,
        "max_time": 0,
        "p95_time": 0,
        "p99_time": 0,
    }

    logger.info("="*80)
    logger.info(f"测试阶段结束: 并发数 = {concurrency}")
    
    if total_requests > 0:
        stage_results["success_rate"] = (success_count / total_requests) * 100
        stage_results["rps"] = total_requests / duration_seconds
        
        logger.info(f"总请求数: {total_requests}")
        logger.info(f"  - 成功: {success_count} | 失败: {failed_count} | 异常: {exception_count}")
        logger.info(f"成功率: {stage_results['success_rate']:.2f}%")
        logger.info(f"请求/秒 (RPS): {stage_results['rps']:.2f}")

        if latencies:
            stage_results["avg_time"] = np.mean(latencies)
            stage_results["min_time"] = np.min(latencies)
            stage_results["max_time"] = np.max(latencies)
            stage_results["p95_time"] = np.percentile(latencies, 95)
            stage_results["p99_time"] = np.percentile(latencies, 99)
            logger.info("-" * 20 + " 成功请求响应时间 (秒) " + "-" * 20)
            logger.info(f"平均值: {stage_results['avg_time']:.3f} | 最小值: {stage_results['min_time']:.3f} | 最大值: {stage_results['max_time']:.3f}")
            logger.info(f"P95: {stage_results['p95_time']:.3f} | P99: {stage_results['p99_time']:.3f}")
    else:
        logger.warning("此阶段没有完成任何请求。")
    
    logger.info("="*80 + "\n")
    return stage_results


def plot_results(all_results: list[dict], output_filename="pressure_test_results.png"):
    """
    根据所有测试阶段的结果生成图表。
    """
    if not all_results or len(all_results) < 1:
        logger.warning("没有足够的测试数据来生成图表。")
        return

    # 按并发数排序，确保图表X轴有序
    all_results.sort(key=lambda x: x['concurrency'])

    concurrencies = [r['concurrency'] for r in all_results]
    rps_values = [r['rps'] for r in all_results]
    avg_times_ms = [r['avg_time'] * 1000 for r in all_results]
    p95_times_ms = [r['p95_time'] * 1000 for r in all_results]
    p99_times_ms = [r['p99_time'] * 1000 for r in all_results]
    success_rates = [r['success_rate'] for r in all_results]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle('大模型并发压力测试结果', fontsize=16)

    # --- 左 Y 轴: 响应时间 ---
    ax1.set_xlabel('并发数 (Concurrency Level)')
    ax1.set_ylabel('响应时间 (毫秒)', color='tab:blue')
    ax1.plot(concurrencies, avg_times_ms, 'o-', color='tab:blue', label='平均响应时间 (Avg)')
    ax1.plot(concurrencies, p95_times_ms, 's--', color='tab:cyan', label='P95 响应时间')
    ax1.plot(concurrencies, p99_times_ms, '^-', color='tab:purple', label='P99 响应时间')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 右 Y 轴: RPS 和 成功率 ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('RPS & 成功率 (%)', color='tab:red')
    ax2.plot(concurrencies, rps_values, 'd-', color='tab:red', label='RPS (请求/秒)')
    # 为了区分，用条形图表示成功率
    ax2.bar(concurrencies, success_rates, width=np.diff(concurrencies)[0]*0.2 if len(concurrencies)>1 else 0.5, 
            alpha=0.3, color='tab:green', label='成功率 (%)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # 设置成功率Y轴范围
    ax2.set_ylim(0, max(max(rps_values)*1.2 if rps_values else 10, 105))


    # --- 图例 ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局为标题留出空间
    
    # 保存图表
    plt.savefig(output_filename, dpi=300)
    logger.info(f"测试结果图表已保存至: {output_filename}")
    # plt.show() # 如果在GUI环境中运行，可以取消注释以显示图表


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='大模型 API 持续并发压力测试工具',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--concurrency-levels', 
        nargs='+', type=int, default=[2, 4, 8], 
        help='并发数级别列表，例如: 10 20 30 50'
    )
    parser.add_argument(
        '-d', '--duration', 
        type=float, default=0.2, 
        help='每个并发级别持续的时间（分钟）'
    )
    parser.add_argument(
        '--token', 
        type=str, default=DEFAULT_API_TOKEN,
        help='API 的 Bearer Token'
    )
    parser.add_argument(
        '--model',
        type=str, default=DEFAULT_MODEL_NAME,
        help='要测试的大模型名称'
    )
    parser.add_argument(
        '--url',
        type=str, default=DEFAULT_API_URL,
        help='API 请求的 URL'
    )
    args = parser.parse_args()
    
    if args.token == DEFAULT_API_TOKEN:
        logger.warning("="*60)
        logger.warning("警告: 您正在使用默认的 API Token。")
        logger.warning("请使用 --token 参数指定您的真实凭证，否则请求将失败。")
        logger.warning("="*60)
        time.sleep(3)

    # 初始化API处理器
    # 如果要换成其他API，在这里实例化对应的类即可
    api_handler = VolcAPI(
        api_url=args.url,
        api_token=args.token,
        model_name=args.model
    )
    
    # 用于存储所有阶段的结果以供绘图
    all_stage_results = []

    # 依次执行每个测试阶段
    for level in sorted(list(set(args.concurrency_levels))): # 去重并排序
        stage_summary = run_test_stage(
            concurrency=level, 
            duration_minutes=args.duration,
            api_handler=api_handler
        )
        all_stage_results.append(stage_summary)

        if level != args.concurrency_levels[-1]:
            logger.info(f"等待 5 秒后进入下一阶段...")
            time.sleep(5) # 在阶段之间留出一些缓冲时间

    logger.info("所有测试阶段完成！")

    # 生成最终的性能图表
    plot_results(all_stage_results)