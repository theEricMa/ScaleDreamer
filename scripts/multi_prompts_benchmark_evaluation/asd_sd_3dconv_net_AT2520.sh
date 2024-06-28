CUDA_VISIBLE_DEVICES=0  python launch.py \
    --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_50k.yaml \
    --test \
    system.prompt_processor.prompt_library="att3d_2520_prompt_library" \
    system.weights="pretrained/3d_checkpoints/AT2520_3dconv_net.pth"

# # better to run in multi-gpu mode
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_50k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="att3d_2520_prompt_library" \
#    system.weights="pretrained/3d_checkpoints/AT2520_3dconv_net.pth"

