import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from knowledge_tracing_benchmark.core import trace

sequence=[1,1,0,1]
print('Response sequence:', sequence)
print('Mastery trace:', [round(v,3) for v in trace(sequence)])
