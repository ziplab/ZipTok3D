import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SharedDecoderProtocolTests(unittest.TestCase):
    def test_recurrent_sequence_is_selected_state_plus_physical_prefix(self):
        source = (ROOT / "cod/models/vae/networks/decoder.py").read_text(
            encoding="utf-8"
        )
        autoencoder = (ROOT / "cod/models/vae/autoencoder.py").read_text(
            encoding="utf-8"
        )
        config = (ROOT / "config/model/ae_ziptok3d.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("keep_ratio: 0.25", config)
        self.assertIn("sequence = torch.cat([state, z], dim=1)", source)
        self.assertIn("state = updated[:, :selected.size(1)]", source)
        self.assertIn(
            "self._restore_full_tokens(init_tokens, state, indices)", source
        )
        self.assertIn("def _decode_physical_prefixes", autoencoder)
        self.assertIn("z_group = z.index_select(0, indices)[:, :prefix_length]", autoencoder)
        self.assertIn("mask=None", autoencoder)
        self.assertNotIn("num_merged_tokens", config)
        self.assertNotIn("_MergingModule", source)


if __name__ == "__main__":
    unittest.main()
