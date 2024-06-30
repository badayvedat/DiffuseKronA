import subprocess
import time


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def add_embeddings(embedding_path, pipe):
    from safetensors.torch import load_file

    class embedding:
        tokens = ["<s0>", "<s1>"]

    embedding_state_dict = load_file(embedding_path)
    textual_inversions = []

    print(embedding_state_dict.keys())
    if "clip_l" in embedding_state_dict:
        text_encoder_key = "clip_l"
    elif "text_encoders_0" in embedding_state_dict:
        text_encoder_key = "text_encoders_0"
    else:
        raise ValueError(
            "Invalid embedding state dict. Needs to have a key that is either 'clip_l' or 'text_encoders_0'"
        )

    if "clip_g" in embedding_state_dict:
        text_encoder_2_key = "clip_g"
    elif "text_encoders_1" in embedding_state_dict:
        text_encoder_2_key = "text_encoders_1"
    else:
        raise ValueError(
            "Invalid embedding state dict. Needs to have a key that is either 'clip_g' or 'text_encoders_1'"
        )

    for te_key, text_encoder_ref, tokenizer in [
        (text_encoder_key, pipe.text_encoder, pipe.tokenizer),
        (text_encoder_2_key, pipe.text_encoder_2, pipe.tokenizer_2),
    ]:
        if te_key in embedding_state_dict:
            print(f"Loading textual inversion for {te_key} with {embedding.tokens}")
            textual_inversions.append((embedding.tokens, text_encoder_ref, tokenizer))
            pipe.load_textual_inversion(
                embedding_state_dict[te_key],
                token=embedding.tokens,
                text_encoder=text_encoder_ref,
                tokenizer=tokenizer,
            )

    return pipe


def read_json_file(path: str):
    import json

    with open(path) as fp:
        return json.load(fp)


def replace_tokens(prompt: str, token_map: dict[str, str]) -> str:
    for k, v in token_map.items():
        prompt = prompt.replace(k, v)
    return prompt
