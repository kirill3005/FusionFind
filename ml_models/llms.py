from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

class Qwen_model:
    def __init__(self, use = False):
        self.use = use
        if not use:
            return


        self.model_ru = Qwen2VLForConditionalGeneration.from_pretrained(
            "2Vasabi/tvl-mini-0.1", torch_dtype=torch.float16, device_map="auto"
        )

        print('Non russian language set. Switching to standart Qwen2VL model')
        self.model_en = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen2-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

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
                     "text": "You are an ordinary person. Your task is to describe the image in as much detail as possible, but in no more than 150 words, paying maximum attention to all details"},
                ],
            }
        ]
        return messages

    @staticmethod
    def format_instruction_captioning_ru(image_base64):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": 'data:image;base64,' + image_base64,
                    },
                    {"type": "text",
                     "text": "Ты обычный человек. Твоя задача макисмально подробно описать изображение, но не более чем на 150 слов, максимально обращая внимание на все делали"},
                ],
            }
        ]
        return messages

    def create_caption(self, image, lang='ru'):
        if lang=='ru':
            prompt = self.format_instruction_captioning_ru(image)
        else:
            prompt = self.format_instruction_captioning_en(image)
        if not self.use:
            return ''
        text = self.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompt)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        if lang=='ru':
            generated_ids = self.model_ru.generate(**inputs, max_new_tokens=150)
        else:
            print('Non russian language set. Switching to standart Qwen2VL model')
            generated_ids = self.model_ru.generate(**inputs, max_new_tokens=150)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class E5_model:

    def __init__(self, use=False):
        if not use:
            return
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    def get_text_embedding(self, text):
        embeding = self.model.encode(text)
        return embeding