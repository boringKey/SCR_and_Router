## Analysis of Routing Distribution

### Figure: Expert Selection Weights across Blocks (Final Task)

![Routing Distribution](./routing_distribution.png)

### Analysis & Key Findings

We appreciate the reviewer's keen observation regarding the potential for "routing collapse" in MoE architectures. This repository provides empirical evidence demonstrating that our Plasticity Pathway effectively maintains balanced expert utilization without requiring an explicit auxiliary load-balancing loss.

As stated in our manuscript's Appendix B (Experimental Details), the gating mechanism for the two task-specific experts utilizes Noisy Top-k gating. By injecting tunable Gaussian noise into the router logits prior to the Softmax activation, we ensure adequate expert exploration during training. This inherent design effectively prevents the network from lazily converging on a single expert.

To validate this, the visualization above illustrates the routing distribution between Task-Specific Expert 1 ($E_{s1}$) and Task-Specific Expert 2 ($E_{s2}$) across all Transformer layers during Task 4 (the final task) of the EuroSAT dataset.

As shown in the figure, both task-specific experts are actively utilized across all Transformer layers. There is no instance of a single expert being entirely starved, nor is there a global collapse where the network exclusively relies on one expert. This distribution confirms that both experts actively collaborate to process the data, demonstrating that the Noisy Top-k gating successfully maintains a healthy routing balance and empirically ruling out the occurrence of "routing collapse."
