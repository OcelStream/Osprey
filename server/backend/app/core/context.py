import sys
sys.path.append("deepstream/app")
from deepstream import DynamicRTSPPipeline
from backend.app.core.settings import settings

pipeline = DynamicRTSPPipeline(settings=settings)
