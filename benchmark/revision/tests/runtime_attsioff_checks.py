"""Audit upstream model behavior without training or changing its architecture."""
import importlib.util
import json
import sys

import torch

spec = importlib.util.spec_from_file_location("upstream_attsioff", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
torch.manual_seed(0)
model = module.RNAFM_SIPRED_2(dp=0.1, device="cpu").eval()
sizes = {"sirna_gibbs_energy": 20, "pssm_score": 1, "gc_sterch": 1,
         "sirna_second_percent": 2, "sirna_second_energy": 1,
         "tri_nt_percent": 64, "di_nt_percent": 16, "single_nt_percent": 4,
         "gc_content": 1}
batch = {name: torch.rand(4, width, 1) for name, width in sizes.items()}
batch["rnafm_encode"] = torch.rand(4, 21, 640)
batch["rnafm_encode_mrna"] = torch.rand(4, 59, 640)
order = torch.tensor([3, 2, 1, 0])
with torch.no_grad():
    original = model(batch)
    permuted = model({k: v[order] for k, v in batch.items()})[order]
    singleton = torch.cat([model({k: v[i:i+1] for k, v in batch.items()}) for i in range(4)])
    augmented = dict(batch, **{k: torch.full((4, 1), 1000.) for k in ["s-biopredsi", "dsir", "i-score"]})
    augmented_predictions = model(augmented)
    positional = model.pos_embed(torch.zeros(21, 4, 16))
result = {
    "torch": torch.__version__,
    "untrained_model_seed": 0,
    "predictions": original.tolist(),
    "permutation_max_absolute_difference": (original-permuted).abs().max().item(),
    "singleton_max_absolute_difference": (original-singleton).abs().max().item(),
    "optional_scores_max_absolute_difference": (original-augmented_predictions).abs().max().item(),
    "position_encoding_changes_across_nucleotides": not torch.equal(positional[0], positional[1]),
    "position_encoding_changes_across_samples": not torch.equal(positional[:, 0], positional[:, 1]),
}
assert result["permutation_max_absolute_difference"] > 1e-6
assert result["optional_scores_max_absolute_difference"] == 0
assert not result["position_encoding_changes_across_nucleotides"]
assert result["position_encoding_changes_across_samples"]
def sequence_positions(self, x):
    return x + self.pe[:, :x.size(0)].transpose(0, 1)
module.PositionalEncoding.forward = sequence_positions
with torch.no_grad():
    corrected = model(batch)
    corrected_permuted = model({k: v[order] for k, v in batch.items()})[order]
result["corrected_permutation_max_absolute_difference"] = (corrected-corrected_permuted).abs().max().item()
assert result["corrected_permutation_max_absolute_difference"] < 1e-6
print(json.dumps(result, indent=2))
