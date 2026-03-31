### Table 1

| **Dataset**           | **Setting** | **LwF (Last. ↑)** | **iCaRL (Last. ↑)** | **Observation** |
| --------------------- | ----------- | ----------------- | ------------------- | --------------- |
| **ImageNet100** ([1]) | B10-10      | **72.40%**        | 70.96%              | **LwF leads**   |
| **ImageNet100** ([1]) | B50-10      | 72.22%            | **73.96%**          | iCaRL leads     |
| **ImageNet-R** (Ours) | B100-10     | 78.22%            | **78.79%**          | iCaRL leads     |
| **ImageNet-R** (Ours) | B100-20     | **73.12%**        | 72.95%              | **LwF leads**   |

**Table 1: Performance comparison of LwF and iCaRL using CLIP-based implementations.** “BX-Y” denotes $X$ base classes followed by $Y$ incremental tasks. Utilizing the validated reproduction framework from ZSCL [1], our results show that **relative performance varies across different benchmarks and settings**, explaining why iCaRL does not consistently outperform LwF. [1] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models, ICCV 2023.

### Figure 1

![Routing Distribution](./routing_distribution.png)

**Figure 1: Expert Selection Weights for Final Task on EuroSAT.** Both Task-Specific Experts are actively utilized across all layers, demonstrating a healthy routing balance without expert starvation. This is achieved via our Noisy Top-K gating mechanism (Appendix B), which injects tunable Gaussian noise into router logits to ensure expert exploration. Consequently, our approach prevents "routing collapse" without requiring an explicit auxiliary loss.
