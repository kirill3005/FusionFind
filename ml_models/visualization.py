from PIL import Image
from llms import Qwen_model
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict
from diffusers import StableDiffusion3InpaintPipeline
from diffusers import SD3Transformer2DModel


from segment_anything import build_sam, SamPredictor

import torch


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

class Qwen_visualize(Qwen_model):

    @staticmethod
    def format_instruction_captioning_en(image_base64):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": 'data:image;base64,' + image_base64,
                    },
                    {"type": "text",
                     "text": "Your task is to give me a list of clothes in the image. You also must describe every item, that you give. Separate items through new line"},
                ],
            }
        ]
        return messages


class Visualization:

    def __init__(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))


        ckpt_id = "stabilityai/stable-diffusion-3.5-large"
        ckpt_4bit_id = "sayakpaul/sd35-large-nf4"

        transformer_4bit = SD3Transformer2DModel.from_pretrained(
            ckpt_4bit_id, subfolder="transformer", torch_dtype=torch.float16
        )
        self.pipeline = StableDiffusion3InpaintPipeline.from_pretrained(
            ckpt_id,
            transformer=transformer_4bit,
            torch_dtype=torch.float16,
        )
        self.pipeline.enable_model_cpu_offload()

        self.captioner = Qwen_visualize()

    def inpaint_image(self, image_path, text, prompt):
        target_object = self.captioner.create_caption(text, lang='en')
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=target_object,
            box_threshold=0.3,
            text_threshold=0.25
        )

        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])

        self.sam_predictor.set_image(image_source)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        image_mask = masks[0][0].cpu().numpy()
        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(image_mask)
        image_source_for_inpaint = image_source_pil.resize((512, 512))
        image_mask_for_inpaint = image_mask_pil.resize((512, 512))
        image_inpainting = self.pipeline(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint,
                                    num_inference_steps=50).images[0]
        image_inpainting = image_inpainting.resize((image_source_pil.size[0], image_source_pil.size[1]))
        return image_inpainting
