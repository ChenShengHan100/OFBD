# Code provenance

OFBD incorporates and adapts the training framework, dataset loaders, residual
backbones, balanced contrastive objective, logit adjustment, and augmentation
utilities from [ConCutMix](https://github.com/PanHaulin/ConCutMix), by Haolin Pan,
Yong Guo, Mianjie Yu, and Jian Chen. The upstream revision is
`f0e42f5` (the original commits are retained in this repository's history).

These implementations are included directly in OFBD. The OFBD model and loss
APIs use OFBD names; renaming does not change their origin. OFBD extends this
foundation with foreground/background aggregation, local-energy region mixing,
and a residual reinforcement-learning region selector.

```bibtex
@article{pan2024enhanced,
  title={Enhanced Long-Tailed Recognition with Contrastive CutMix Augmentation},
  author={Pan, Haolin and Guo, Yong and Yu, Mianjie and Chen, Jian},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```

Optional external comparison implementations are recorded in
[`third_party/BASELINE_PROVENANCE.md`](third_party/BASELINE_PROVENANCE.md).
Their authorship and license files remain with their original repositories.

## Licensing status

The imported ConCutMix revision does not contain a license file. This release
does not assign a new project-wide license to that upstream code. Public source
availability and the OFBD API names do not grant additional rights to third-party
code. Any project-wide license must account for the upstream permissions.
