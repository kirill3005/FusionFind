from classification import Text_classification_en, Text_classification_ru, Image_classification
import re
class Security_pipeline:
    def __init__(self):
        self.model_multilang = Text_classification_en()
        self.model_ru = Text_classification_ru()
        self.model_images = Image_classification()
        self.regular_expr = r'[^а-яА-Яa-zA-Z\s.,!?;:\']'

    def sanitize_input(self, input_text):
        return re.sub(self.regular_expr, '', input_text)

    @staticmethod
    def count_russian_and_english_chars(text):
        russian_count = 0
        non_russian_count = 0

        for char in text:
            if 'а' <= char <= 'я' or 'А' <= char <= 'Я':
                russian_count += 1
            else:
                non_russian_count += 1


        if russian_count>=non_russian_count:
            return True
        else:
            return False

    def classify_text(self, text):
        is_russian = self.count_russian_and_english_chars(text)
        if is_russian:
            cls = self.model_ru.classify(text)
        else:
            cls = self.model_multilang.classify(text)
        if cls:
            return True
        else:
            return False

    def classify_image(self, image):
        cls = self.model_images.classifier(image)

        if cls:
            return True
        else:
            return False

    def pipeline(self, text, image=None):
        text_clear = self.sanitize_input(text)
        text_cls = self.classify_text(text_clear)

        if not text_cls:
            return False
        if image:
            image_cls = self.classify_image(image)
        else:
            return True

        if not image_cls:
            return False
        else:
            return True