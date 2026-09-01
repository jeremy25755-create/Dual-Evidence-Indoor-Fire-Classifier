# Fire Dual Semantic Evidence Lite

A lightweight three-class indoor-fire classifier built from:

1. a 32-dimensional level encoder,
2. a 32-dimensional multi-scale trend encoder, and
3. separate Level and Trend Dirichlet semantic evidence heads, and
4. uncertainty-derived confidence fusion.

The model does not use TEFN BPA modules or a temporal persistence gate. It
classifies Background, Fire, and Nuisance. Uncertainty is derived from total
class evidence rather than learned as a fourth class.

```powershell
& ".\.venv\Scripts\python.exe" ".\Fire_Semantic_Evidence_Lite\train.py" `
  --data ".\dataset\Indoor Fire Dataset with Distributed Multi-Sensor Nodes.csv" `
  --seq-len 60 `
  --stride 10 `
  --embedding-dim 32 `
  --evidence-hidden-dim 64 `
  --dropout 0.1 `
  --epochs 30 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --kl-weight 0.01 `
  --branch-loss-weight 0.2 `
  --annealing-epochs 10 `
  --patience 5 `
  --runs 1 `
  --seed 2026 `
  --device cuda `
  --output ".\results_fire_dual_semantic_evidence_lite"
```
