# Third-party augmentation baselines

These optional comparison repositories are pinned as Git submodules. The active
OFBD training entry points do not import them. Retrieve them with
`git submodule update --init --recursive`; consult each repository for its
dependencies and license. The older `main_1_thirdparty_aug.py` adapter is not part
of this release.

| Label | Snapshot | Status | Core source used |
|---|---|---|---|
| SaliencyMix | `SaliencyMix` @ `9ee02b8` | author release | `SaliencyMix_CIFAR/saliencymix.py::saliency_bbox` |
| SnapMix | `SnapMix` @ `245bda4` | author release | `utils/mixmethod.py::get_spm,snapmix` |
| PuzzleMix | `PuzzleMix` @ `e2dbf3a` | author release + compatible fallback | `mixup.py::mixup_graph`; graph cut requires missing `gco` |
| ResizeMix | `ResizeMix` @ `45441ff` | author release | `run_apis/trainer.py` ResizeMix branch |
| Attentive CutMix | `attentive_cutmix` @ `55e133e` | community implementation | `attentive_transform.py` |
| KeepAugment | `KeepAugment_Pytorch` @ `7982e44` | community implementation | `util/keep_cutout.py::Keep_Cutout` |

Attentive CutMix and KeepAugment must be reported as community implementations.

Additional attention references are pinned at:

| Reference | Snapshot | Source |
|---|---|---|
| Coordinate Attention | `CoordAttention` @ `7619bea` | https://github.com/houqb/CoordAttention |
| CBAM / BAM | `attention-module` @ `459efad` | https://github.com/Jongchan/attention-module |
| Triplet Attention | `triplet-attention` @ `2cf02fb` | https://github.com/landskape-ai/triplet-attention |

The `.gitmodules` file records the upstream URLs for every snapshot. Third-party
names identify their original methods; they are not renamed as OFBD components.
