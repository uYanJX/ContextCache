from vcache.context.offline_inference import PassthroughInferenceEngine
from vcache.inference_engine.inference_engine import InferenceEngine
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine

try:
    from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
except ImportError:
    OpenAIInferenceEngine = None  # type: ignore[assignment,misc]

try:
    from vcache.inference_engine.strategies.lang_chain import LangChainInferenceEngine
except ImportError:
    LangChainInferenceEngine = None  # type: ignore[assignment,misc]

try:
    from vcache.inference_engine.strategies.vllm import VLLMInferenceEngine
except ImportError:
    VLLMInferenceEngine = None  # type: ignore[assignment,misc]

__all__ = [
    "InferenceEngine",
    "PassthroughInferenceEngine",
    "BenchmarkInferenceEngine",
    "OpenAIInferenceEngine",
    "LangChainInferenceEngine",
    "VLLMInferenceEngine",
]
