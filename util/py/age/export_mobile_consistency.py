"""Export a selected consistency candidate and fail on numerical divergence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ab.nn.nn.MobileAgeNet import Net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--converter', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    if not result['candidate_available']:
        raise RuntimeError('Training produced no eligible candidate')
    config = json.loads((args.run / 'config.json').read_text())
    model = Net((1, 3, 224, 224), (1,), config['model_parameters'], torch.device('cpu'))
    model.load_state_dict(torch.load(args.run / 'best_candidate.pth', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    torch.set_num_threads(4)
    onnx_path, tflite_path = args.run / 'candidate.onnx', args.run / 'candidate.tflite'
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx_path),
                          input_names=['input'], output_names=['output'], opset_version=17, dynamo=False)
    spec = importlib.util.spec_from_file_location('age_converter', args.converter)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    converter._convert_onnx_to_tflite(str(onnx_path), str(tflite_path), str(args.run / 'conversion'))
    converter._verify_exported_tflite_shape(str(tflite_path), 224, 224)
    import onnxruntime as ort
    import tensorflow as tf
    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    lite = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    lite.allocate_tensors()
    inp, out = lite.get_input_details()[0], lite.get_output_details()[0]
    if inp['dtype'] != np.float32 or out['dtype'] != np.float32:
        raise ValueError('This deployment export must remain float32')
    from ab.nn.loader.utkface import loader
    from torchvision.transforms import functional as TF
    def preprocess(image):
        return TF.normalize(TF.to_tensor(image.resize((224, 224), Image.Resampling.BILINEAR)),
                            (.485, .456, .406), (.229, .224, .225))
    _, _, _, validation = loader(lambda _: preprocess, 'age-regression')
    rows = []
    for index in range(min(32, len(validation))):
        item = validation.dataset.dataset[validation.indices[index]]
        x = preprocess(item['image'].convert('RGB')).numpy()[None]
        with torch.inference_mode():
            pt = float(model(torch.from_numpy(x)).item())
        onnx = float(ort_session.run(None, {'input': x})[0].reshape(-1)[0])
        lite.set_tensor(inp['index'], x.transpose(0, 2, 3, 1).copy())
        lite.invoke()
        tflite = float(lite.get_tensor(out['index']).reshape(-1)[0])
        rows.append({'age': float(item['age']), 'pytorch': pt, 'onnx': onnx, 'tflite': tflite})
    repeats = []
    for _ in range(10):
        lite.invoke()
        repeats.append(float(lite.get_tensor(out['index']).reshape(-1)[0]))
    errors = np.asarray([[abs(row['onnx'] - row['pytorch']), abs(row['tflite'] - row['pytorch'])] for row in rows])
    report = {'samples': rows, 'mae_difference': errors.mean(0).tolist(),
              'max_difference': errors.max(0).tolist(), 'tflite_ten_repeats': repeats,
              'tflite_sha256': hashlib.sha256(tflite_path.read_bytes()).hexdigest(),
              'input_shape': inp['shape'].tolist(), 'output_shape': out['shape'].tolist(),
              'input_dtype': str(inp['dtype']), 'output_dtype': str(out['dtype']),
              'input_quantization': inp['quantization'], 'output_quantization': out['quantization'],
              'tensorflow': tf.__version__, 'onnxruntime': ort.__version__,
              'pass': bool(errors.max() < .02 and max(repeats) - min(repeats) < 1e-5)}
    (args.run / 'export_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
    if not report['pass']:
        raise RuntimeError('Export parity/determinism check failed; do not deploy')


if __name__ == '__main__':
    main()
