## Analysis of Routing Distribution

### Figure: Expert Selection Weights across Blocks (Task 4)

![Routing Distribution](/routing_distribution.png) 
*(Note: Please convert your .pdf to .png for direct display in GitHub README, or provide a link to the PDF file)*

### Analysis & Key Findings

To address concerns regarding **"Routing Collapse"** (the tendency of MoE systems to converge on a single expert), we visualized the mean routing weights for two experts ($w1$ and $w2$) across all 12 blocks ($B0$ to $B11$) during Task 4. 

**Key Observations:**

1. **Dynamic Expert Preference:** There is a clear shift in expert utilization across the hierarchical layers. 
   - **Early Blocks (B0-B2):** The router strongly prefers **Expert 1 ($w1$)**, with weights exceeding 0.7.
   - **Middle/Late Blocks (B4-B6, B8-B10):** The preference shifts significantly toward **Expert 2 ($w2$)**, reaching a peak of ~0.72 in Block 8.
   - **Final Block (B11):** The router reverts to a preference for $w1$.

2. **Natural Load Balancing:** Even without an explicit load-balancing loss, the hierarchical routing mechanism demonstrates a high degree of specialization. The system effectively "routes" information to different experts based on the block's position in the hierarchy, rather than collapsing into a single path.

3. **Conclusion:** This empirical evidence confirms that our architecture maintains expert diversity and achieves functional specialization across sequential blocks, effectively preventing routing collapse.
