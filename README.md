# Unsloth Puzzle Submissions 🚀

This repository contains my solutions to Unsloth's machine learning puzzles. I'm currently working on these challenges targeting the **ML Intern** role.

## 📌 Submissions

At this point, I've only had time to complete puzzles B, C, and E. 

For problem B, I demonstrate FSDP1 and FSDP2 with the same loss curves. I also added FSDP2 with gradient accumulation, in case VRAM limitations began to be an issue.

I ran into some troubles with C and discovered I could not compile the bnb section without reworking their code using a Triton kernel. I'm not particularly versed on that, so I didn't attempt it. As such, there are a couple graph breaks caused by this. I limited it to only three breaks by chunking the calculations together.

Problem E successfully computes equivalent gradients to a standard approach, and achieves a 78% reduction in peak VRAM usage for a moderately sized input. It also can successfully compute a large input, by dynamically calculating an appropriate chunk size. I also demonstrate equivalence between my implementation's loss on LLaMa-1b (unsloth/Llama-3.2-1B-Instruct) and standard loss.

- **Google Colab Notebook (Solutions C and E)**: [Link to Colab](https://colab.research.google.com/drive/1rIUf9SZWHGoFiCpt2YbrDPwfbVNUaoZZ?usp=sharing)  
- **Kaggle Notebook (Solution B)**: [Link to Kaggle](https://www.kaggle.com/code/liuserr/notebook2313628fdd)  

## 🔍 About My Approach

These notebooks showcase my problem-solving skills in ML, optimization techniques, and efficient model implementation. I emphasize **VRAM-efficient computation**, **scalability**, and **performance tuning**—critical factors for real-world AI applications.

Thanks for reviewing my puzzle solutions! I enjoyed working on these.
