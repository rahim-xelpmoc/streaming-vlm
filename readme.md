# StreamingVLM: Real-Time Understanding for Infinite Video Streams ⚡️

<p align="center">
    <a href=""><b>Paper</b></a> |
    <a href=""><b>Slides</b></a> |
    <a href="https://streamingvlm.hanlab.ai"><b>Demo Page</b></a>
</p>

## 🧠 TL;DR

StreamingVLM enables real-time, stable understanding of effectively infinite video by keeping a compact KV cache and aligning training with streaming inference. It avoids quadratic cost and sliding-window pitfalls, runs up to 8 FPS on a single H100, and wins 66.18% vs GPT-4o mini on a new long-video benchmark. It also boosts general VQA without task-specific finetuning.
*You can grasp the gist by skimming this section first.*

## 🎬 Demo

Go to [streamingvlm.hanlab.ai](https://streamingvlm.hanlab.ai) to see more cases and try our model.

[https://github.com/user-attachments/assets/1a15b496-55c5-4c66-809d-a49d70e5d864](https://github.com/user-attachments/assets/1a15b496-55c5-4c66-809d-a49d70e5d864)


## 🛠️ Install

```bash
./scripts/env_infer.sh
./scripts/env_sft.sh
```

*You can set up the environment by running the scripts above.*

## 🚀 Inference

<p align="center">
  <img src="figures/infer.png" width="95%" />
</p>

*You can run inference by the command below.*

```bash
conda activate streamingvlm-infer
python streaming_vlm/inference/inference.py
```

## 📚 SFT

### Prepare Dataset

First, download `mit-han-lab/Inf-Stream-Train` to `/path/to/your/Inf-Stream-Train`.
Then, download `chenjoya/Live-WhisperX-526K` to `/path/to/your/Inf-Stream-Train/Livecc_sft`.
Preprocess the LiveCC dataset with the following command:

```bash
cd $DATASET_PATH/Livecc_sft
find . -type f -exec mv -t . {} +
```

Download `mit-han-lab/Inf-Stream-Eval` to `/path/to/your/Inf-Stream-Eval`.

Finally, set environment paths:

```bash
export DATASET_PATH=/path/to/your/Inf-Stream-Train
export EVAL_DATASET_PATH=/path/to/your/Inf-Stream-Eval
```

*You can prepare data by following the steps in order.*

### ▶️ Run SFT

<p align="center">
  <img src="figures/train.png" width="65%" />
</p>
*You can kick off SFT by executing the scripts below.*

```bash
conda activate streamingvlm-sft
./scripts/sft_stage_1.sh
./scripts/sft_stage_2.sh # High Quality Annealing Data
```

## 📊 Evaluation

### Efficiency

```bash
conda activate streamingvlm-infer
./scripts/eval_efficiency.sh  
```

<p align="center">
  <img src="figures/efficiency.png" width="65%" />
</p>

*You can benchmark efficiency by running the script above.*

### OVOBench

First, make the OVOBench data structure like:

```
data/ovobench
├── AutoEvalMetaData
├── COIN
├── cross_task
├── Ego4D
├── hirest
├── MovieNet
├── OpenEQA
├── ovo_bench_new.json
├── perception_test
├── star
├── thumos
├── youcook2
└── YouTube_Games
```

Then, prepare the OVOBench environment and run evaluation:

```bash
./scripts/env_ovo.sh
conda activate streamingvlm-ovo
./scripts/eval_OVOBench.sh
```

*You can start OVOBench eval by these commands.*

### VQA

We use VLMEvalKit to evaluate VQA tasks.

```bash
conda activate streamingvlm-infer
./scripts/eval_VQA.sh
```

*You can launch VQA evaluation with the script above.*

### Inf-Stream-Eval

```bash
conda activate streamingvlm-infer
./scripts/eval_Inf-Stream-Eval.sh
```

*You can run the in-house eval by calling this script.*

### LiveSports3k-cc

```bash
conda activate streamingvlm-infer
export LIVESPORTS3K_PATH=/path/to/your/LiveSports-3K/videos
conda activate streamingvlm-infer
./scripts/eval_LiveSports3k-cc.sh
```

*You can evaluate LiveSports3k-cc with the path set above.*

### Modify FPS

If you would like to change inference FPS, use the following command:

```bash
sed -i 's/^FPS = .*/FPS = float(os.environ.get("QWENVL_FPS", "2.0"))/' \
  "$(python -c 'import inspect,qwen_vl_utils.vision_process as m; import os; print(os.path.abspath(inspect.getsourcefile(m)))')"
```

*You can tweak FPS by editing the line via the command above.*
