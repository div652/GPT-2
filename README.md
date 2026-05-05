<img width="1600" height="900" alt="better-language-models" src="https://github.com/user-attachments/assets/ac25dadd-4e04-48b5-bcc3-d1bd3e2d4121" />

# GPT-2
Implementing GPT2 from Scratch

The goal for this repo to achieve the following 
- Implementing the whole GPT-2 architecture in code using pytorch
- Able to load weights from the hugging face GPT-2 implementation, to make sure the models match
- Training our own GPT-2 using GPUs on a 10B tokens dataset
- Bringing down the runtime per forward pass so that we efficiently use GPUs and train fast.
- We use the following optimisations,
- - Using FlashAttention
- - Using tensorfloat32 -> bfloat16
  - Using torch.compile()
  - Using DistributedDataParallel technqiues
- We later switch to a more refined data set : FineWebTune.

## Resources Used to Refer
- Andrej Karparthy's video :   https://www.youtube.com/watch?v=l8pRSuU81PU&t=1802s
- Andrej Karparthy's repo :   https://github.com/karpathy/build-nanogpt
- Attention is all you need paper : https://arxiv.org/abs/1706.03762
- OpenAI GPT-2 Paper : https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT2 Blogpost : https://openai.com/index/better-language-models/
- OpenAI GPT-3 Paper :https://arxiv.org/pdf/2005.14165 
