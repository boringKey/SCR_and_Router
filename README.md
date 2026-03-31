### Table 1

| **Dataset**           | **Setting** | **LwF (Last. ↑)** | **iCaRL (Last. ↑)** | **Observation** |
| --------------------- | ----------- | ----------------- | ------------------- | --------------- |
| **CIFAR100** ([1])    | B10-10      | 48.04%            | **70.97%**          | iCaRL leads     |
| **ImageNet100** ([1]) | B10-10      | **72.40%**        | 70.96%              | **LwF leads**   |
| **ImageNet100** ([1]) | B50-10      | 72.22%            | **73.96%**          | iCaRL leads     |
| **ImageNet-R** (Ours) | B100-10     | 78.22%            | **78.79%**          | iCaRL leads     |
| **ImageNet-R** (Ours) | B100-20     | **73.12%**        | 72.95%              | **LwF leads**   |

**Table 1: Performance comparison between LwF and iCaRL under multi-modal adaptations.** “BX-Y” denotes a base task of $X$ classes followed by $Y$ tasks. Following ZSCL [1] protocols, our results (including ImageNet-R) show that relative performance in VLMs varies across different benchmarks and settings, explaining why iCaRL’s advantage over LwF is not uniform. [1] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models, ICCV 2023.

### Figure 1

![Routing Distribution](/Users/kangborui/研究生/typoraImage/routing_distribution-4938740.png)

**Figure 1: Expert Selection Weights for Task 4 on EuroSAT.** Both Task-Specific Experts ($E_{s1}, E_{s2}$) are actively utilized across all layers, demonstrating a healthy routing balance without expert starvation. This is achieved via our Noisy Top-$k$ gating mechanism (Appendix B), which injects tunable Gaussian noise into router logits to ensure expert exploration. Consequently, our approach prevents "routing collapse" without requiring an explicit auxiliary loss.
