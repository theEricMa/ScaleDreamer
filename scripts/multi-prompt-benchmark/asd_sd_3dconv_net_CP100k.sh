# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_500k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="cap3d_100k_prompt_library" \

# better to run in multi-gpu mode
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
    --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_500k.yaml \
    --train \
    system.prompt_processor.prompt_library="cap3d_100k_prompt_library" \
