import dataclasses

import torch.linalg
import torch.nn

from jlens import JacobianLens


@dataclasses.dataclass(frozen=True, kw_only=True)
class LensLayerComparison:
    relative_frobenius_distance: float
    """Symmetric relative Frobenius distance"""

    cos_similarity: float
    """Cosine similarity of the flattened matrices"""

    spectral_distance: float
    """Spectral norm distance"""


def compare(lens_a: JacobianLens, lens_b: JacobianLens) -> dict[int, LensLayerComparison]:
    assert (lens_a.source_layers, lens_a.d_model) == (lens_b.source_layers, lens_b.d_model)

    comparisons = {}
    for layer in lens_a.source_layers:
        diff_frob_norm = torch.linalg.matrix_norm(lens_a.jacobians[layer] - lens_b.jacobians[layer])
        frob_norm_a = torch.linalg.matrix_norm(lens_a.jacobians[layer])
        frob_norm_b = torch.linalg.matrix_norm(lens_b.jacobians[layer])

        assert frob_norm_a > 0, f"lens_a has a zero matrix at layer {layer}"
        assert frob_norm_b > 0, f"lens_b has a zero matrix at layer {layer}"

        cos_sim = torch.nn.functional.cosine_similarity(lens_a.jacobians[layer].flatten(), lens_b.jacobians[layer].flatten(), dim=0)

        spectral_distance = torch.linalg.matrix_norm(lens_a.jacobians[layer] - lens_b.jacobians[layer], ord=2)

        comparisons[layer] = LensLayerComparison(
            relative_frobenius_distance=(2 * diff_frob_norm / (frob_norm_a + frob_norm_b)).item(),
            cos_similarity=cos_sim.item(),
            spectral_distance=spectral_distance.item()
        )

    return comparisons
