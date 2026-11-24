"""Load the saved CPU model and suggest a genre for a local audio file."""
import argparse
import json
from time import perf_counter

import torch

from .audio import ROOT, prepare


class Predictor:
    def __init__(self):
        torch.set_num_threads(2)
        self.config = json.loads((ROOT / 'results' / 'model_config.json').read_text(encoding='utf-8'))
        self.model = torch.jit.load(str(ROOT / 'results' / 'genre_cnn.pt'), map_location='cpu').eval()
        with torch.inference_mode():
            for _ in range(20):
                self.model(torch.zeros(1, 1, 64, 216))

    def predict(self, path):
        start = perf_counter()
        feature, timing = prepare(path)
        forward_start = perf_counter()
        with torch.inference_mode():
            probabilities = self.model(feature.unsqueeze(0)).softmax(1)[0]
        timing['forward_ms'] = 1000 * (perf_counter() - forward_start)
        confidence, index = probabilities.max(0)
        value = float(confidence)
        result = {'suggested_genre': self.config['genres'][int(index)], 'confidence': value,
                  'status': 'auto_tag' if value >= self.config['threshold'] else 'needs_review',
                  'threshold': self.config['threshold'],
                  'scores': dict(zip(self.config['genres'], probabilities.tolist())), **timing}
        result['total_ms'] = 1000 * (perf_counter() - start)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    args = parser.parse_args()
    print(json.dumps(Predictor().predict(args.audio), indent=2))
