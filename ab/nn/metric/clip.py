import torch

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install the 'transformers' library for CLIP metric: pip install transformers")

MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model_cache = {}


def _get_clip_model(device):
    if 'model' not in _clip_model_cache:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _clip_model_cache['model'] = model
        _clip_model_cache['processor'] = processor
    _clip_model_cache['model'].to(device)
    return _clip_model_cache['model'], _clip_model_cache['processor']


class CLIPMetric:
    def __init__(self, out_shape=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.similarity_scores = []
        self.num_samples = 0

    def __call__(self, preds, labels):
        """
        This method now expects `preds` to be a tuple of
        (generated_image_tensors, raw_text_prompts)
        from the model's hijacked forward pass.
        """
        if not isinstance(preds, (list, tuple)) or len(preds) != 2:
            return

        generated_images, text_prompts = preds

        # Ensure the lists are not empty
        if len(generated_images) == 0 or len(text_prompts) == 0:
            return

        model, processor = _get_clip_model(self.device)
        model.eval()

        with torch.no_grad():
            # The CLIP processor can handle raw tensors and text directly
            inputs = processor(
                text=text_prompts, images=generated_images, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = model(**inputs)

            # Calculate cosine similarity
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

            # The logits_per_image are the cosine similarities
            batch_scores = outputs.logits_per_image.diag()  # Get the score for each (image, text) pair

            self.similarity_scores.append(batch_scores.sum().item())
            self.num_samples += len(generated_images)

    def result(self):
        if self.num_samples == 0:
            return 0.0
        # The CLIP score is typically scaled by 100
        avg_score=sum(self.similarity_scores) / self.num_samples /100
        return avg_score

    def get_all(self):
        return {'CLIP_Score': self.result()}


def create_metric(out_shape=None, device=None):
    return CLIPMetric(out_shape=out_shape, device=device)