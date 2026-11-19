import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from ziptok_retrain.model import WeightedPrefixAutoencoder
except ImportError:
    torch = None


class TrainedConfigTests(unittest.TestCase):
    def test_singleelt_model_and_solver_match_frozen_training_settings(self):
        model = (ROOT / "config/model/ae_singleelt.yaml").read_text()
        solver = (ROOT / "config/solver/ae_singleelt.yaml").read_text()
        train = (ROOT / "config/train_singleelt_shapenet.yaml").read_text()

        for setting in (
            "ziptok_retrain.model.WeightedPrefixAutoencoder",
            "num_latents: 128",
            "use_learnable_pos: true",
            "num_layers: 12",
            "num_loops: 6",
            "num_merged_tokens: 128",
            "nested_dropout_strategy: exp",
        ):
            self.assertIn(setting, model)
        for setting in (
            "ziptok_retrain.solver.MultiBudgetSolver",
            "lr: 3.0e-5",
            "coeff_intermediate: 0.5",
            "val_budgets: [1, 2, 4, 128]",
            "score_budgets: [1, 2, 4]",
        ):
            self.assertIn(setting, solver)
        self.assertIn("model_selection_split: val", train)
        self.assertIn("accumulate_grad_batches: 7", train)

    @unittest.skipUnless(torch is not None, "requires torch and pointops")
    def test_weight_shapes_and_physical_prefixes(self):
        model = WeightedPrefixAutoencoder(
            output_dim=1,
            num_latents=8,
            embed_dim=32,
            query_dim=4,
            use_learnable_pos=True,
            nested_dropout_strategy="exp",
            encoder_params={
                "num_patches": 4,
                "num_blocks": 1,
                "num_layers_per_block": 1,
                "causal_self_attention": False,
            },
            decoder_params={
                "output_resolution": 8,
                "output_patch_size": 4,
                "num_layers": 2,
                "num_init_layers": 1,
                "num_loops": 3,
                "num_merged_tokens": 8,
                "keep_ratio": 0.25,
            },
        )
        self.assertEqual(tuple(model.latent_pos.shape), (8, 32))
        self.assertEqual(tuple(model.decoder.merging_module.tokens.shape), (8, 32))
        self.assertEqual(len(model.decoder.transformer.transformer), 2)
        self.assertEqual(model.decoder.num_loops, 3)

        lengths = []

        def record_prefix(z, **kwargs):
            lengths.append(z.shape[1])
            planes = torch.zeros(z.shape[0], 3, 4, 8, 8)
            return planes, planes, planes

        model.eval()
        with mock.patch.object(model.decoder, "decode", side_effect=record_prefix):
            result = model.decode(
                torch.zeros(2, 8, 32),
                active_num_latents=torch.tensor([1, 8]),
            )
        self.assertEqual(lengths, [1, 8])
        self.assertEqual(tuple(result[0].shape), (2, 3, 4, 8, 8))


if __name__ == "__main__":
    unittest.main()
