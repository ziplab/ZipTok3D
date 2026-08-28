import unittest

import torch

from cod.models.diffusion.stage2_edm import Attention, SABERStage2EDM


class CausalStage2EDMTests(unittest.TestCase):
    def test_causal_attention_prefix_is_invariant_to_suffix(self):
        torch.manual_seed(7)
        attention = Attention(
            query_dim=8,
            heads=2,
            head_dim=4,
            dropout=0.0,
            causal=True,
        ).eval()
        prefix = torch.randn(2, 2, 8)
        full = torch.cat([prefix, torch.randn(2, 3, 8)], dim=1)

        prefix_output = attention(prefix)
        full_output = attention(full)[:, :prefix.size(1)]

        torch.testing.assert_close(prefix_output, full_output)

    def test_sampling_uses_requested_physical_prefix_length(self):
        model = SABERStage2EDM(
            num_latents=4,
            channels=4,
            num_classes=3,
            width=8,
            num_heads=2,
            head_dim=4,
            depth=1,
            time_channels=8,
            gradient_checkpointing=False,
        ).eval()
        labels = torch.tensor([0, 1])

        samples = model.sample(labels, num_steps=2, num_latents=2)

        self.assertEqual(tuple(samples.shape), (2, 2, 4))
        self.assertTrue(all(block.self_attn.causal for block in model.model.blocks))
        self.assertTrue(all(not block.cross_attn.causal for block in model.model.blocks))

    def test_full_edm_prefix_is_invariant_to_suffix(self):
        torch.manual_seed(11)
        model = SABERStage2EDM(
            num_latents=4,
            channels=4,
            num_classes=3,
            width=8,
            num_heads=2,
            head_dim=4,
            depth=2,
            time_channels=8,
            gradient_checkpointing=False,
        ).eval()
        with torch.no_grad():
            model.model.proj_out.weight.normal_()
        prefix = torch.randn(2, 2, 4)
        full = torch.cat([prefix, torch.randn(2, 2, 4)], dim=1)
        sigma = torch.tensor([0.5, 1.0])
        labels = torch.tensor([0, 1])

        prefix_output = model(prefix, sigma, labels)
        full_output = model(full, sigma, labels)[:, :prefix.size(1)]

        torch.testing.assert_close(prefix_output, full_output)

    def test_sampling_rejects_prefix_above_training_maximum(self):
        model = SABERStage2EDM(
            num_latents=4,
            channels=4,
            num_classes=3,
            width=8,
            num_heads=2,
            head_dim=4,
            depth=1,
            time_channels=8,
            gradient_checkpointing=False,
        )

        with self.assertRaises(ValueError):
            model.sample(torch.tensor([0]), num_latents=5)


if __name__ == "__main__":
    unittest.main()
