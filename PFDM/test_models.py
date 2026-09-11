from __future__ import annotations

import unittest

import torch

from PFDM.models import PPGFormerEncoder, PhysioCycleEncoding


class PhysioCycleEncodingTest(unittest.TestCase):
    def test_rejects_cycle_range_above_sequence_nyquist(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            PhysioCycleEncoding(
                channels=32,
                sample_rate=20.0,
                downsample_factor=16,
                min_bpm=50.0,
                max_bpm=150.0,
            )

    def test_ppg_encoder_applies_cycle_encoding_before_downsampling(self) -> None:
        model = PPGFormerEncoder(
            channels=8,
            embedding_size=16,
            transformer_layers=1,
            transformer_heads=4,
            dropout=0.0,
            sample_rate=20.0,
            use_frequency_branch=True,
        ).eval()
        cycle_input_shapes: list[tuple[int, ...]] = []
        handle = model.cycle.register_forward_pre_hook(
            lambda _module, inputs: cycle_input_shapes.append(tuple(inputs[0].shape))
        )
        try:
            with torch.no_grad():
                output = model(torch.zeros(2, 3000, 1))
        finally:
            handle.remove()

        self.assertEqual(cycle_input_shapes, [(2, 3000, 8)])
        self.assertEqual(output["embedding"].shape, (2, 188, 16))
        self.assertEqual(output["tokens"].shape, (2, 188, 16))
        self.assertEqual(model.cycle.downsample_factor, 1)


if __name__ == "__main__":
    unittest.main()
