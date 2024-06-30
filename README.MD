# DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Model

An unofficial implementation of [DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Model paper](https://arxiv.org/abs/2402.17412)

I am not able to reproduce their experiments yet, the quality is significantly
worse than LoRA. Probably there is a bug somewhere in my implementation or
I missed out some details on paper.

## Training
```bash
python train.py \
    --input_images train-images.zip \
    --output_dir output-models
```


## Inference
```bash
python inference.py \
    --prompt 'photo of TOK' \
    --krona_path output-models/krona.safetensors \
    --embedding_path output-models/embeddings.pti \
    --special_params_path output-models/special_params.json \
    --num_images 4
```

