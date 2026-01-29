"""
Demo script to launch the vLLM-Omni Orchestrator with a UniversalStage
and send a sample request.
"""

import os
import asyncio

# 设置日志级别，确保能看到详细日志
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_LOGGING_PREFIX", "[UniversalTest]")

async def run_test():
    # 1. Initialize the Omni Orchestrator with our custom universal stage config
    # We use a dummy model name since the universal stage doesn't load weights from a model dir by default.
    model_name = "dummy-universal-model"
    config_path = os.path.join(os.path.dirname(__file__), "universal_stage_config.yaml")

    print("--- Script started ---", flush=True)
    print(f"--- Current working directory: {os.getcwd()} ---", flush=True)
    print(f"--- Python path includes vllm-omni: {'vllm_omni' in str(os.sys.path)} ---", flush=True)
    print(f"--- Config path: {config_path} ---", flush=True)
    print(f"--- Config exists: {os.path.exists(config_path)} ---", flush=True)

    from vllm_omni.entrypoints.omni import Omni
    import torch

    print("--- Checking torch availability ---", flush=True)
    print(f"--- Torch version: {torch.__version__} ---", flush=True)
    print(f"--- CUDA available: {torch.cuda.is_available()} ---", flush=True)

    # 检查 vllm_omni 模块是否能正确导入
    print("--- Checking vllm_omni imports ---", flush=True)
    try:
        from vllm_omni.universal.universal_stage import UniversalStage
        print("--- UniversalStage import successful ---", flush=True)
    except Exception as e:
        print(f"--- UniversalStage import failed: {e} ---", flush=True)
        import traceback
        traceback.print_exc()
        return

    print("--- Initializing Omni ---", flush=True)
    try:
        omni_engine = Omni(
            model=model_name,
            stage_configs_path=config_path,
            stage_init_timeout=10,
            init_timeout=20,
            batch_timeout=2,
        )
        print("--- Omni initialized successfully ---", flush=True)
    except Exception as exc:
        print(f"--- Omni initialization failed: {exc} ---", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # 2. Prepare multiple dummy multi-modal requests
    # Since our vision/audio operators just log the input type and return it, 
    # we can send a simple tensor or a dictionary.
    num_requests = 5  # 发送 5 个请求
    request_ids = [f"test-req-{i:03d}" for i in range(num_requests)]
    dummy_inputs = [torch.randn(1, 10, 512) for _ in range(num_requests)]
    
    print(f"--- Sending {num_requests} requests: {request_ids} ---", flush=True)

    # Import SamplingParams (from vllm if available, else mock)
    try:
        from vllm.sampling_params import SamplingParams
    except ImportError:
        class SamplingParams:
            def __init__(self, temperature=1.0, max_tokens=1, **kwargs):
                self.temperature = temperature
                self.max_tokens = max_tokens
                for k, v in kwargs.items():
                    setattr(self, k, v)

    # Match the config's default_sampling_params
    # Note: sampling_params_list should be per-stage, not per-request
    # For single stage, we need only 1 sampling params object
    sampling_params = SamplingParams(temperature=1.0, max_tokens=1)

    # generate() is the main entrypoint for the orchestrator
    # It routes the request through the stages defined in the yaml
    try:
        results = omni_engine.generate(
            prompts=dummy_inputs,
            sampling_params_list=[sampling_params]  # Per-stage params, not per-request
        )
    except Exception as exc:
        print(f"--- Generate failed: {exc} ---", flush=True)
        raise

    # 3. Process and print results
    print(f"--- Received {len(results)} results ---", flush=True)
    for i, output in enumerate(results):
        print(f"Request {output.request_id} (index {i}) finished: {output.finished}")
        print(f"Output stage: {output.stage_id}")
        
        # In UniversalStage, we mapped outputs to multimodal_output
        if output.multimodal_output:
            for k, v in output.multimodal_output.items():
                if isinstance(v, torch.Tensor):
                    print(f"  - Modality: {k}, Tensor Shape: {v.shape}")
                else:
                    print(f"  - Modality: {k}, Value: {v}")
        print()  # 空行分隔每个请求的结果

    print("--- Test Complete ---", flush=True)

if __name__ == "__main__":
    asyncio.run(run_test())
