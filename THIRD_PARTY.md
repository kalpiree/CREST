# Sources and baseline adaptations

Model weights and datasets are acquired separately. Their access conditions and licenses remain those of their publishers. Installing this package does not grant access to gated checkpoints or change the terms of external resources.

| Component | Source | Implementation used here |
| --- | --- | --- |
| Spotlighting | [Hines et al., 2024](https://arxiv.org/abs/2403.14720) | Caret datamarking of record field contents and an accompanying system instruction. Records and candidates are preserved. |
| Prompt Guard 2--Record | [Meta model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | The frozen 86M classifier, overlapping token chunks, maximum score across chunks and appearances, and identity-wide record removal. |
| RewriteDetection-Record | [Wang et al., 2025](https://arxiv.org/abs/2409.11690) | Continuation n-gram overlap and recommendation-frequency change, followed by record removal while retaining the associated candidate item. |
| RETURN-Del | [Ning et al., 2025](https://arxiv.org/abs/2504.02458), [RETURN source](https://github.com/Biglemon-Ning/RETURN/tree/f56bc959890a39a8eae83097d041e33026c2495e) | Sparse position-gap co-occurrence counts and the upstream row-normalized support calculation. This adaptation uses deletion without replacement or ensemble voting. |
| TextSimu attack adaptation | [Wang et al., 2025](https://arxiv.org/abs/2409.11690) | Reference retrieval, keyword extraction, persona drafts, and feedback-driven rewriting applied to inference-time item-text records. Qwen2.5 replaces the original attack generator and reference encoder. |

These are the recommendation-record adaptations evaluated by CREST. They do not reproduce the original papers' benchmark settings or numerical results.

## Checkpoints

| Model | Revision |
| --- | --- |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c702b33eeacc393d103063234e8bc28` |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `a8ded8e697ce7c355e395a0df51f94adb4a2fd27` |

Prompt Guard's generic binary labels are resolved using Meta's [reference implementation](https://github.com/meta-llama/llama-cookbook/blob/3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded/getting-started/responsible_ai/prompt_guard/inference.py). The adapter validates the pinned checkpoint before applying that mapping.

Qwen2.5's last-token output-head optimization checks the [Transformers 4.44.2 implementation](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py). Qwen3.5 uses the native text model in [Transformers 5.3.0](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py). Both preserve the constrained candidate-label output format; their environments and caches are separate.

## Data

The evaluation uses Amazon Reviews 2018 All Beauty, the Steam review and item data, and MovieLens-1M. Use the preparation commands in the README to acquire and process the required files. Preserve each publisher's attribution and distribution restrictions when working with the downloaded data.
