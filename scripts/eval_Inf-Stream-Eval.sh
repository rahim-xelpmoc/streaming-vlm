# Generate
    # StreamingVLM
./streaming_vlm/eval/model_compete/generate.sh -m mit-han-lab/StreamingVLM -b Qwen # infinite stream
    # LiveCC
./streaming_vlm/eval/model_compete/generate.sh -m chenjoya/LiveCC-7B-Instruct -b LiveCC -d 100 -t 0.7  # chunk 100 seconds
./streaming_vlm/eval/model_compete/generate.sh -m chenjoya/LiveCC-7B-Instruct -b LiveCC -t 0.7 # infinite stream
    # Qwen2.5-VL-7B-Instruct
./streaming_vlm/eval/model_compete/generate.sh -m Qwen/Qwen2.5-VL-7B-Instruct -b Qwen -d 100 -t 0.7 # chunk 100 seconds
./streaming_vlm/eval/model_compete/generate.sh -m Qwen/Qwen2.5-VL-7B-Instruct -b Qwen -t 0.7 # infinite stream

# Merge
./streaming_vlm/eval/model_compete/merge.sh mit-han-lab/StreamingVLM
./streaming_vlm/eval/model_compete/merge.sh chenjoya/LiveCC-7B-Instruct
./streaming_vlm/eval/model_compete/merge.sh Qwen/Qwen2.5-VL-7B-Instruct

# Score
./streaming_vlm/eval/model_compete/score.sh --model1 "mit-han-lab/StreamingVLM" --model2 "gpt-4o-mini" 
./streaming_vlm/eval/model_compete/score.sh --model1 "chenjoya/LiveCC-7B-Instruct" --model2 "gpt-4o-mini" 
./streaming_vlm/eval/model_compete/score.sh --model1 "Qwen/Qwen2.5-VL-7B-Instruct" --model2 "gpt-4o-mini" 
