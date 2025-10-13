export EVAL_DATASET_PATH=/path/to/your/livesports3k_eval_dataset

# predict
python streaming_vlm/eval/livesports3kcc/distributed_generate_streaming.py  --model_name_or_path mit-han-lab/StreamingVLM --output_dir results/livesports3k/streamingvlm --num_workers 8 --temperature 0.9 --repetition_penalty 1.05
# judge
python streaming_vlm/eval/livesports3kcc/llm_judge.py --model_id StreamingVLM --prediction_jsonl results/livesports3k/streamingvlm/StreamingVLM.jsonl --output_dir results/livesports3kcc/judges --num_workers 64 --baseline_id LLaVA-Video-72B-Qwen2 --baseline_jsonl streaming_vlm/eval/livesports3kcc/captions/LLaVA-Video-72B-Qwen2.jsonl
# other options
# python streaming_vlm/eval/livesports3kcc/llm_judge.py --model_id StreamingVLM --prediction_jsonl results/livesports3k/streamingvlm/StreamingVLM.jsonl --output_dir results/livesports3kcc/judges --num_workers 64 --baseline_id Gemini-1.5-pro --baseline_jsonl streaming_vlm/eval/livesports3kcc/captions/Gemini-1.5-pro.jsonl
# python streaming_vlm/eval/livesports3kcc/llm_judge.py --model_id StreamingVLM --prediction_jsonl results/livesports3k/streamingvlm/StreamingVLM.jsonl --output_dir results/livesports3kcc/judges --num_workers 64 --baseline_id GPT-4o --baseline_jsonl streaming_vlm/eval/livesports3kcc/captions/GPT-4o.jsonl
# python streaming_vlm/eval/livesports3kcc/llm_judge.py --model_id StreamingVLM --prediction_jsonl results/livesports3k/streamingvlm/StreamingVLM.jsonl --output_dir results/livesports3kcc/judges --num_workers 64 --baseline_id Livecc --baseline_jsonl streaming_vlm/eval/livesports3kcc/livecc/LiveCC-7B-Instruct.jsonl
