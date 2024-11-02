from transformers import pipeline


class Text_classification_ru:

    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="cointegrated/rubert-base-cased-nli-threeway'")
        self.labels = ["шоппинг", "не шоппинг"]
    def classify(self, text):
        probs = self.classifier(text, self.labels)

        if probs[0] > 0.5:
            return True
        else:
            return False

class Text_classification_en:

    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.labels = ["shopping", "not shopping"]
    def classify(self, text):
        probs = self.classifier(text, self.labels)

        if probs[0] > 0.5:
            return True
        else:
            return False

class Image_classification:

    def __init__(self):
        self.classifier = pipeline(model='openai/clip-vit-large-patch14', task="zero-shot-image-classification")
        self.labels = ["shopping", "not shopping"]
    def classify(self, image):
        probs = self.classifier(image, self.labels)

        if probs[0]['score'] > 0.5:
            return True
        else:
            return False

