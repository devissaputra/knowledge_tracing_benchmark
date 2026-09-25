import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from knowledge_tracing_benchmark.core import predictive_trace,trace

sequence=[1,1,0,1]
print("Synthetic software fixture only; not an empirical result.")
print("Response sequence:",sequence)
print("Pre-response probabilities:",[round(v,3) for v in predictive_trace(sequence)])
print("Post-response mastery:",[round(v,3) for v in trace(sequence)])
