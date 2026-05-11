<img width="1600" height="900" alt="better-language-models" src="https://github.com/user-attachments/assets/ac25dadd-4e04-48b5-bcc3-d1bd3e2d4121" />

# <img height="40" alt="openai-logo-png" src="https://github.com/user-attachments/assets/a04f82be-094a-4153-8517-7f8561c7b943" /> GPT-2 : From Scratch 


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

### Checkpoint 1 :
Finished the basic version of the code.
Model is training on the gpu, but I am using a small batch size of 4, since the RAM is limited. 
I achieved loss upto 5.2 on the data. 
The training curve looks like this : 

<img height="200" alt="Normal_train" src="https://github.com/user-attachments/assets/24fe79e5-2771-466b-9ebe-bc866949dea8" />

Tok/sec : 3650 
Time per forward pass : 1120 ms. 

### Checkpoint 2:
Moved to Jarvis Labs. 
Models suddenly trains so much faster. 
epoch : 0, step : 100, loss : 6.503375053405762, dt: 294.09ms , tok/sec: 13927.59. Batch_size =4 


Shifted to batch_size = 16 
epoch : 0, step : 51, loss : 6.358279228210449, dt: 1121.38ms , tok/sec: 14610.57


## Checkpoint 3 : Using TF-lite 32. 
Batch_size = 16 
use 'high' predicistoin 
epoch : 0, step : 38, loss : 6.3488264083862305, dt: 404.91ms , tok/sec: 40463.02

Achieved LOSS Below 5 .After 1000 steps almost. 
epoch : 0, step : 1104, loss : 4.933437347412109, dt: 407.97ms , tok/sec: 40159.87

<img height="213" alt="image" src="https://github.com/user-attachments/assets/505303dc-6a58-4e16-ac19-a60307a4c009" />



## Resources Used to Refer
![Check out the video](https://img.youtube.com/vi/l8pRSuU81PU/0.jpg)([https://youtube.com](https://www.youtube.com/watch?v=l8pRSuU81PU))
- Andrej Karparthy's video :   https://www.youtube.com/watch?v=l8pRSuU81PU&t=1802s
- Andrej Karparthy's repo :   https://github.com/karpathy/build-nanogpt
- Attention is all you need paper : https://arxiv.org/abs/1706.03762
- OpenAI GPT-2 Paper : https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT2 Blogpost : https://openai.com/index/better-language-models/
- OpenAI GPT-3 Paper :https://arxiv.org/pdf/2005.14165 
