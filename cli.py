import argparse


def get_train_args():
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument(
        "--input_images",
        type=str,
        required=True,
        help="A .zip or .tar file containing the image files that will be used for fine-tuning",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time.",
    )
    train_parser.add_argument(
        "--resolution",
        default=1024,
        type=int,
        help="Square pixel resolution which your images will be resized to for training",
    )
    train_parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size (per device) for training",
    )
    train_parser.add_argument(
        "--num_train_epochs",
        default=4000,
        type=int,
        help="Number of epochs to loop through your training dataset",
    )
    train_parser.add_argument(
        "--max_train_steps",
        default=1000,
        type=int,
        help="Number of individual training steps. Takes precedence over num_train_epochs",
    )
    train_parser.add_argument(
        "--ti_lr",
        default=3e-4,
        type=float,
        help="Scaling of learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
    )
    train_parser.add_argument(
        "--krona_lr",
        default=5e-4,
        type=float,
        help="Scaling of learning rate for training LoRA embeddings. Don't alter unless you know what you're doing.",
    )
    train_parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="Learning rate scheduler to use for training",
    )
    train_parser.add_argument(
        "--lr_warmup_steps",
        default=100,
        type=int,
        help="Number of warmup steps for lr schedulers with warmups.",
    )
    train_parser.add_argument(
        "--token_string",
        default="TOK",
        type=str,
        help="A unique string that will be trained to refer to the concept in the input images. Can be anything, but TOK works well",
    )
    train_parser.add_argument(
        "--caption_prefix",
        type=str,
        default="a photo of TOK, ",
        help="Text which will be used as prefix during automatic captioning. Must contain the `token_string`. For example, if caption text is 'a photo of TOK', automatic captioning will expand to 'a photo of TOK under a bridge', 'a photo of TOK holding a cup', etc.",
    )
    train_parser.add_argument(
        "--mask_target_prompts",
        type=str,
        default=None,
        help="Prompt that describes part of the image that you will find important. For example, if you are fine-tuning your pet, `photo of a dog` will be a good prompt. Prompt-based masking is used to focus the fine-tuning process on the important/salient parts of the image",
    )
    train_parser.add_argument(
        "--crop_based_on_salience",
        type=bool,
        default=True,
        help="If you want to crop the image to `target_size` based on the important parts of the image, set this to True. If you want to crop the image based on face detection, set this to False",
    )
    train_parser.add_argument(
        "--use_face_detection_instead",
        type=bool,
        default=False,
        help="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
    )
    train_parser.add_argument(
        "--clipseg_temperature",
        type=float,
        default=1.0,
        help="How blurry you want the CLIPSeg mask to be. We recommend this value be something between `0.5` to `1.0`. If you want to have more sharp mask (but thus more errorful) you can decrease this value.",
    )
    train_parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=999999,
        help="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
    )
    train_parser.add_argument(
        "--input_images_filetype",
        type=str,
        choices=["zip", "tar", "infer"],
        help="Filetype of the input images. Can be either `zip` or `tar`. By default its `infer`, and it will be inferred from the ext of input file.",
    )
    train_parser.add_argument(
        "--verbose", type=bool, default=True, help="verbose output"
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for training images",
    )
    return train_parser.parse_args()


def get_inference_args():
    inference_args = argparse.ArgumentParser()
    inference_args.add_argument(
        "--prompt", required=True, type=str, help="Prompt to generate image from"
    )
    inference_args.add_argument(
        "--negative_prompt",
        default="",
        type=str,
        help="Negative prompt to generate image from",
    )
    inference_args.add_argument(
        "--num_inference_steps", default=30, type=int, help="Number of inference steps"
    )
    inference_args.add_argument("--seed", default=None, type=int, help="Generator seed")
    inference_args.add_argument(
        "--num_images", default=1, type=int, help="Number of images to generate"
    )
    inference_args.add_argument(
        "--output_path",
        default="output_images",
        type=str,
        help="Output path for generated image",
    )
    inference_args.add_argument(
        "--krona_path", required=True, type=str, help="Path to krona weights"
    )
    inference_args.add_argument(
        "--krona_scale", type=float, default=1.0, help="Krona scale"
    )
    inference_args.add_argument(
        "--embedding_path", required=True, type=str, help="Path to embeddings"
    )
    inference_args.add_argument(
        "--special_params_path",
        required=True,
        type=str,
        help="Path to special params json file",
    )
    inference_args.add_argument(
        "--scheduler",
        default="K_EULER",
        type=str,
        help="Scheduler to use",
        choices=[
            "DDIM",
            "DPMSolverMultistep",
            "HeunDiscrete",
            "KarrasDPM",
            "K_EULER_ANCESTRAL",
            "K_EULER",
            "PNDM",
        ],
    )
    return inference_args.parse_args()
