# To Reviewer NZ5u

 This repository is prepared for rebuttal and serves three purposes:

 - ##### Answer to “Why does iCaRL not always outperform LwF?”
   
   - This section addresses the reviewer's question about the relative ranking of **iCaRL** and **LwF**.
   
 - ##### Answer to “Q2/Q3 codebase questions”
   
   - This section addresses the reviewer's **Q2/Q3 codebase questions**.
   
 - ##### Answer to “Limitation: Routing Collapse”
   
   - This section addresses the **limitation discussion on** **Routing Collapse**.

## Answer to “Why does iCaRL not always outperform LwF?”

To avoid any misunderstanding, we would like to elaborate on this performance dynamics in detail by breaking it down into two key factors.

First, regarding implementation, we adopt the multi-modal adaptations of these methods from ZSCL [1] to ensure a rigorous and traceable baseline. Since the original iCaRL and LwF were designed for single-modal scenarios, we leverage the ZSCL as it provides established reproductions of these classic methods specifically adapted for the CLIP architecture.

Second, empirical evidence suggests that performance gains across diverse VLM benchmarks do not follow a uniform pattern. As established in ZSCL (see Table below), iCaRL’s lead on CIFAR100 vanishes or even reverses on ImageNet100.

| Datasets (from [1]) | Setting | LwF (Last. ↑) | iCaRL (Last. ↑) | Observation   |
| :------------------ | :------ | :------------ | :-------------- | :------------ |
| **CIFAR100**        | B10-10  | 48.04%        | **70.97%**      | iCaRL leads   |
| **ImageNet100**     | B10-10  | **72.40%**    | 70.96%          | **LwF leads** |
| **ImageNet100**     | B50-10  | 72.22%        | **73.96%**      | iCaRL leads   |

*("BX-Y" denotes a base task of X classes followed by a sequence totaling Y tasks)*

Furthermore, we conduct supplemental experiments on **ImageNet-R** and observed the same phenomenon across different task counts. While iCaRL performs better with 10 tasks (B100-10), LwF surpasses iCaRL in Last accuracy when the task count increases to 20 (B100-20).

| Dataset        | Setting | LwF (Last. ↑) | iCaRL (Last. ↑) | Observation   |
| :------------- | :------ | :------------ | :-------------- | :------------ |
| **ImageNet-R** | B100-10 | 78.22%        | **78.79%**      | iCaRL leads   |
| **ImageNet-R** | B100-20 | **73.12%**    | 72.95%          | **LwF leads** |

These findings demonstrate that when adapted to VLMs, the relative rankings of these baselines naturally fluctuate depending on the dataset and task setting. Our results are consistent with these established empirical trends in the field.

**Reference:**
[1] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models, ICCV 2023.


 ## **Answer to “Q2/Q3 codebase questions”**

 For ease of inspection, the `SCR_calculate/` folder only contains the minimum files required to compute **SCR**.

 **Which file contains SCR in the code files?**

 The SCR computation is implemented in **`SCR_calculate/calculate_SCR.py`**.

 **Relevant line numbers:**

 - **`SCR_calculate/calculate_SCR.py:71-110`**
   - Function `compute_scores_with_matrix(...)`
   - This is the core implementation that computes the task-wise SCR score from:
     - the zero-shot performance matrix, and
     - the task-to-upstream similarity matrix.
     - The task-to-upstream similarity matrix is computed by **`SCR_calculate/calculate_sim.py`**.

 - **`SCR_calculate/calculate_SCR.py:217-220`**
   - This part aggregates the task-wise scores and prints the final overall **SCR** value.

 - **`SCR_calculate/calculate_SCR.py:222-234`**
   - This part additionally computes and prints the low-, mid-, and high-similarity grouped SCR values.

 ## **Answer to “Limitation: Routing Collapse”**

 ### Figure: Expert Selection Weights across Blocks (Final Task)

 ![Routing Distribution](/Users/kangborui/研究生/typoraImage/routing_distribution-4927975.png)

 ### Analysis & Key Findings

 We appreciate the reviewer's keen observation regarding the potential for "routing collapse" in MoE architectures. This repository provides empirical evidence demonstrating that our Plasticity Pathway effectively maintains balanced expert utilization without requiring an explicit auxiliary load-balancing loss.

 As stated in our manuscript's Appendix B (Experimental Details), the gating mechanism for the two task-specific experts utilizes Noisy Top-k gating. By injecting tunable Gaussian noise into the router logits prior to the Softmax activation, we ensure adequate expert exploration during training. This inherent design effectively prevents the network from lazily converging on a single expert.

 To validate this, the visualization above illustrates the routing distribution between Task-Specific Expert 1 ($E_{s1}$) and Task-Specific Expert 2 ($E_{s2}$) across all Transformer layers during Task 4 (the final task) of the EuroSAT dataset.

 As shown in the figure, both task-specific experts are actively utilized across all Transformer layers. There is no instance of a single expert being entirely starved, nor is there a global collapse where the network exclusively relies on one expert. This distribution confirms that both experts actively collaborate to process the data, demonstrating that the Noisy Top-k gating successfully maintains a healthy routing balance and empirically ruling out the occurrence of "routing collapse."
