import torch   
import os
import hashlib

def hash_prompt(model: str, prompt: str, type: str) -> str:
    identifier = f"{model}-{prompt}-{type}"
    return hashlib.md5(identifier.encode()).hexdigest()


def load_from_cache(cache_dir, pretrained_model_name_or_path, prompt, load_local=False, load_global=False, ):
        # load global text embedding
        if load_global:
            cache_path_global = os.path.join(
                cache_dir,
                f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
            )
            if not os.path.exists(cache_path_global):
                raise FileNotFoundError(
                    f"Global Text embedding file {cache_path_global} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
                )
            global_text_embedding = torch.load(cache_path_global, map_location='cpu')

        # load local text embedding
        if load_local:
            cache_path_local = os.path.join(
                cache_dir,
                f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
            )
            if not os.path.exists(cache_path_local):
                raise FileNotFoundError(
                    f"Local Text embedding file {cache_path_local} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
                )
            local_text_embedding = torch.load(cache_path_local, map_location='cpu')

        # the return value depends on the flags
        if load_local and load_global:
            return global_text_embedding, local_text_embedding
        elif load_local:
            return local_text_embedding
        elif load_global:
            return global_text_embedding

def _load_prompt_embedding(args):
    """
        Load the global/local text embeddings for a single prompt
        from cache into memory
    """
    prompt, prompt_vds, cache_dir, pretrained_model_name_or_path = args
    global_text_embeddings, local_text_embeddings = load_from_cache(
        cache_dir, pretrained_model_name_or_path, prompt, 
        load_global=True, load_local=True
    )
    text_embeddings_vd = torch.stack(
        [load_from_cache(
            cache_dir, pretrained_model_name_or_path, prompt, 
            load_global=False, load_local=True) for prompt in prompt_vds], dim=0 # we don't need local text embeddings for view-dipendent conditional generation
    )
    return prompt, global_text_embeddings, local_text_embeddings, text_embeddings_vd