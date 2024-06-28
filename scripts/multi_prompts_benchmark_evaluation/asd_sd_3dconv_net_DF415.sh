CUDA_VISIBLE_DEVICES=0,  python launch.py \
    --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_100k.yaml \
    --test \
    system.prompt_processor.prompt_library="dreamfusion_415_prompt_library" \
    system.weights="pretrained/3d_checkpoints/DF415_3dconv_net.pth"

# # better to run in multi-gpu mode
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_sd_3dconv_net_100k.yaml \
#     --test \
#     system.prompt_processor.prompt_library="dreamfusion_415_prompt_library" \
#     system.weights="pretrained/3d_checkpoints/DF415_3dconv_net.pth"