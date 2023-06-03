import torch
import clip 
from lavis.models import load_model_and_preprocess


from PIL import Image

class BLIPModel(object):

    def __init__(self):

        print("Loading BLIP...")
        self.device = torch.device("cuda")
        # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
        # this also loads the associated image processors
        blip_model, blip_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)

        print("BLIP loaded.")

        self.model = blip_model
        self.processors = blip_processors
        

    def predict(self,image: Image):
        image = image.convert('RGB')
        image = self.processors["eval"](image).unsqueeze(0).to(self.device)

        # generate caption
        result = self.model.generate({"image": image})
        return result[0]


class CLIPModel(object):

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        print("Loading CLIP...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

        print("CLIP loaded.")

        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device

    def predict(self,image, sentences, return_probs=False): # NEEDS REVIEW, UNUSED VARS, CPU?
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(sentences).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if return_probs:
            return probs
        else:
            return sentences[probs.argmax()]
