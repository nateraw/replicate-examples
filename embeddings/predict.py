# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from typing import List, Union
import cog
from cog import BasePredictor, Input
from pathlib import Path
import numpy as np
# Can uncomment to not use fast_sentence_transformers (in cases where onnx doesn't play nice)
# from sentence_transformers import SentenceTransformer
import torch
import onnxruntime as ort
print("Torch cuda available", torch.cuda.is_available())
print("torch cuda.get_device_name(0)", torch.cuda.get_device_name(0))
print("ort.get_available_providers()", ort.get_available_providers())
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
import json
# Can uncomment to download weights from GCP. In that case, the os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" line above is not needed
# from utils import maybe_download_with_pget, Logger
from datasets import load_dataset, Dataset
from datasets.builder import DatasetGenerationError


class Predictor(BasePredictor):
    model_name = "BAAI/bge-large-en-v1.5"
    # gcp_bucket_weights = "gs://bucket-name/sentence-transformers/all-mpnet-base-v2/bd44305fd6a1b43c16baf96765e2ecb20bca8e1d"

    def setup(self):
        # maybe_download_with_pget(
        #     self.model_name,
        #     self.gcp_bucket_weights,
        #     [
        #         "config.json",
        #         "pytorch_model.bin",
        #         "quantized_false.onnx",
        #         "special_tokens_map.json",
        #         "tokenizer_config.json",
        #         "tokenizer.model",
        #         "vocab.txt",
        #         "config_sentence_transformers.json",
        #         "sentence_bert_config.json",
        #     ],
        #     Logger(),
        # )
        self.model = SentenceTransformer(self.model_name)

    def predict(
        self,
        path: cog.Path = Input(description="Path to file containing text as JSONL with 'text' field or valid JSON string list.", default=None),
        texts: str = Input(description='text to embed, formatted as JSON list of strings (e.g. ["hello", "world"])', default=""),
        batch_size: int = Input(description="Batch size to use when processing text data.", default=32),
        normalize_embeddings: bool = Input(description="Whether to normalize embeddings.", default=True),
        convert_to_numpy: bool = Input(description="When true, return output as npy file. By default, we return JSON", default=False),
    ) -> Union[List[str], cog.Path]:
        ds = None
        text_field = "text"

        if len(texts):
            texts = json.loads(texts)
            ds = Dataset.from_dict({"text": texts})
        else:
            path = Path(path)
            try:
                def gen_data():
                    for line in Path(path).open():
                        ex = json.loads(line)
                        if text_field not in ex:
                            raise ValueError(f"JSON object must contain a '{text_field}' field")
                        yield ex
                ds = Dataset.from_generator(gen_data)
            except DatasetGenerationError:
                print("Failed to load as jsonl, trying as if it's a JSON list of strings")
                ds = Dataset.from_dict({"text": json.loads(path.read_text())})
        if ds is None:
            raise ValueError("Must provide either a `path` to a txt file containing a JSON list of strings OR `texts`, a JSON list of strings")
        
        # Without using datasets library, you can do this (can exclude the 'datasets' dependency in cog.yaml and the import above if you do this)
        # texts = []
        # for line in path.open():
        #     ex = json.loads(line)
        #     if text_field not in ex:
        #         raise ValueError(f"JSON object must contain a '{text_field}' field")
        #     texts.append(ex[text_field])

        # embeddings = self.model.encode(
        #     texts,
        #     convert_to_numpy=True,
        #     normalize_embeddings=normalize_embeddings,
        #     batch_size=batch_size,
        #     show_progress_bar=True
        # )
        
        def process(ex, text_field=text_field):
            ex["embedding"] = self.model.encode(
                ex[text_field],
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=True
            )
            return ex
        
        ds = ds.map(process, batched=True, batch_size=5000)
        if convert_to_numpy:
            ds.set_format('numpy')
            out_name = '/tmp/embeddings.npy'
            np.save(out_name, ds['embedding'])
            print(f"Saved embeddings to {out_name}. To load, run `np.load('{out_name}')`")
            return cog.Path(out_name)
        else:
            return ds['embedding']


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    out = predictor.predict(path="example.jsonl", texts="", batch_size=64, normalize_embeddings=True, convert_to_numpy=False)
    out2 = predictor.predict(path="example.jsonl", texts="", batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    data2 = np.load(out2)
    print(data2.shape)
    # out = predictor.predict(path='../samsum_as_list.txt', texts="", batch_size=64, normalize_embeddings=True)
    # out = predictor.predict(path=None, texts='["embed this", "also this pls"]', batch_size=64, normalize_embeddings=True, text_field="text")
