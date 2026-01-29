
import uuid
import multiprocessing as mp
from queue import Empty
from typing import List, Sequence, Union, Any, Dict
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.logger import init_logger
from .config import UniversalStageConfig
from .engine import UniversalEngine

logger = init_logger(__name__)

class UniversalStage:
    """Stage implementation for universal operators, mimicking OmniDiffusion architecture.
    
    This stage acts as a generic processor in the multi-modal pipeline, 
    capable of routing data through a sequence of operators.
    """
    
    import multiprocessing as mp

    def __init__(self, config: UniversalStageConfig):
        self.config = config
        self.stage_id = config.stage_id
        self.engine_input_source = config.engine_input_source

        # 解析 runtime 里的 num_workers 和 devices
        runtime = getattr(config, 'runtime', {})
        # 优先使用 num_workers，如果没有则使用 num_gpus
        self.num_workers = int(runtime.get('num_workers', runtime.get('num_gpus', 1)))
        logger.info(f"[Stage-{self.stage_id}] Creating {self.num_workers} workers")
        devices_str = runtime.get('devices', 'cpu')
        if devices_str == 'cpu':
            self.devices = ['cpu'] * self.num_workers
        else:
            # 逗号分割，自动分配
            dev_list = [d.strip() for d in devices_str.split(',') if d.strip()]
            if len(dev_list) < self.num_workers:
                # 不足则重复填充
                dev_list = (dev_list * ((self.num_workers + len(dev_list) - 1) // len(dev_list)))[:self.num_workers]
            self.devices = dev_list

        # 多 worker 支持：每个 worker 一个进程，进程间用队列通信
        self._task_queues = []  # 主进程 -> worker
        self._result_queues = []  # worker -> 主进程
        self._workers = []
        for i in range(self.num_workers):
            worker_config = config
            if hasattr(worker_config, 'runtime'):
                worker_config.runtime = dict(worker_config.runtime)
            else:
                worker_config.runtime = {}
            worker_config.runtime['device'] = self.devices[i]
            worker_config.worker_id = i
            task_q = mp.Queue()
            result_q = mp.Queue()
            p = mp.Process(target=_universal_worker_main, args=(worker_config, task_q, result_q))
            p.daemon = True
            p.start()
            self._task_queues.append(task_q)
            self._result_queues.append(result_q)
            self._workers.append(p)
            logger.info(f"[Stage-{self.stage_id}] Started worker {i} on device {self.devices[i]}")

    def generate(
        self,
        prompts: Union[OmniPromptType, Sequence[OmniPromptType]],
        sampling_params: Any = None,
        request_ids: List[str] = [],
        **kwargs
    ) -> List[OmniRequestOutput]:
        """Generate/process outputs for the given prompts.
        支持多 worker 并发。
        """
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        else:
            prompts = list(prompts)

        # Completion of request_ids logic
        if len(request_ids) < len(prompts):
            new_ids = [f"{i + len(request_ids)}_{uuid.uuid4()}"
                       for i in range(len(prompts) - len(request_ids))]
            request_ids.extend(new_ids)

        # 按 worker 数量分组分发
        n = len(prompts)
        worker_indices = [i % self.num_workers for i in range(n)]
        # 发送任务到各 worker
        for i in range(n):
            task = {
                'prompt': prompts[i],
                'sampling_params': sampling_params,
                'request_id': request_ids[i]
            }
            self._task_queues[worker_indices[i]].put(task)
        # 收集结果
        results = [None] * n
        finished = 0
        while finished < n:
            for idx in range(n):
                if results[idx] is not None:
                    continue
                try:
                    res = self._result_queues[worker_indices[idx]].get(timeout=0.01)
                    results[idx] = res
                    finished += 1
                except Empty:
                    continue
        return results


def _universal_worker_main(config, task_q, result_q):
    # 每个进程独立初始化 engine
    worker_id = getattr(config, 'worker_id', 0)
    stage_id = getattr(config, 'stage_id', 0)
    logger = init_logger(f"UniversalWorker-{stage_id}-{worker_id}")
    logger.info(f"[Worker-{worker_id}] Starting universal worker for stage {stage_id}")
    
    engine = UniversalEngine.make_engine(config)
    logger.info(f"[Worker-{worker_id}] Engine initialized successfully")
    
    while True:
        try:
            task = task_q.get()
        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Error getting task: {e}")
            break
        if task is None:
            logger.info(f"[Worker-{worker_id}] Received shutdown signal")
            break
        
        request_id = task['request_id']
        logger.info(f"[Worker-{worker_id}] Processing request {request_id}")
        
        prompt = task['prompt']
        sampling_params = task['sampling_params']
        
        try:
            engine_output = engine.execute(prompt, sampling_params=sampling_params)
            logger.info(f"[Worker-{worker_id}] Request {request_id} processed successfully")
            
            ro = OmniRequestOutput(
                request_id=request_id,
                stage_id=stage_id,
                finished=True
            )
            if hasattr(engine_output, 'multimodal_outputs') and engine_output.multimodal_outputs:
                ro.multimodal_output.update(engine_output.multimodal_outputs)
            if hasattr(engine_output, 'pooling_output') and engine_output.pooling_output is not None:
                ro.multimodal_output["pooling_output"] = engine_output.pooling_output
            
            result_q.put(ro)
            logger.info(f"[Worker-{worker_id}] Result for request {request_id} sent back")
            
        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Error processing request {request_id}: {e}")
            # 发送错误结果
            error_ro = OmniRequestOutput(
                request_id=request_id,
                stage_id=stage_id,
                finished=False
            )
            result_q.put(error_ro)



    def _run_engine(self, prompts: List[OmniPromptType], sampling_params: Any, 
                    request_ids: List[str]) -> List[OmniRequestOutput]:
        """Execute the engine and wrap results in OmniRequestOutput."""
        results = []
        for prompt_data, rid in zip(prompts, request_ids):
            # The 'prompt_data' is a standard OmniPromptType (PromptType | OmniTokensPrompt | etc.)
            engine_output = self.engine.execute(prompt_data, sampling_params=sampling_params)
            
            # Wrap OmniEngineCoreOutput into OmniRequestOutput.
            # OmniRequestOutput is the standard exchange format between stages in vLLM-Omni.
            ro = OmniRequestOutput(
                request_id=rid,
                stage_id=self.stage_id,
                finished=True
            )
            
            # Transfer multimodal outputs (tensors) to the request output
            if engine_output.multimodal_outputs:
                ro.multimodal_output.update(engine_output.multimodal_outputs)
            
            # Extract pooling output if available
            if engine_output.pooling_output is not None:
                ro.multimodal_output["pooling_output"] = engine_output.pooling_output

            # If the engine produced tokens (unlikely for a universal stage, but possible), 
            # they can be added to ro.request_output if needed.
            
            results.append(ro)
        return results

    def close(self) -> None:
        """Cleanup resources."""
        pass

    def __del__(self):
        try:
            self.close()
        except:
            pass
