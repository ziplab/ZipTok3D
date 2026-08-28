# Third-Party Components

This supplementary package includes source from the following research code
bases. Each component remains subject to its own license.

| Component | Location | Upstream | Distribution status | Use |
|---|---|---|---|---|
| 3DShape2VecSet | `3DShape2VecSet-master/` | https://github.com/1zb/3DShape2VecSet | MIT | ShapeNet preprocessing conventions and baseline experiments |
| 3DILG | `3DILG-master/` | https://github.com/1zb/3DILG | CC BY-NC-SA 4.0 | Baseline experiments |
| pointops | External dependency; not included | https://github.com/Silverster98/pointops | Not redistributed in this package | CUDA farthest-point sampling and KNN |

The ZipTok3D implementation builds on the public COD-VAE method and follows
its tokenizer, triplane selection, and occupancy-decoding conventions. The
submission package contains the integrated implementation needed for the paper
pipeline rather than a second, nested COD-VAE checkout.

The pointops upstream repository does not currently state redistribution
terms. This package therefore references it only as an external runtime
dependency and does not include a copy of its source.
