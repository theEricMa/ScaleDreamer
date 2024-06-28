# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv_triplane_transformer_10k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="instant3d_17000_prompt_library"

# better to run in multi-gpu mode
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
    --config configs/multi-prompt_benchmark/asd_mv_triplane_transformer_10k.yaml \
    --train \
    system.prompt_processor.prompt_library="instant3d_17000_prompt_library"
