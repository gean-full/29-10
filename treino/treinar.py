"""Treina o YOLO nano da fábrica e exporta para o navegador (ONNX + rotulos.json em ../modelos/fabrica/).

Uso:  python treinar.py PASTA_ANOTADA [--epocas 150] [--tamanho 416]
PASTA_ANOTADA tem images/ e labels/ (formato YOLO, mesma ordem de classes de classes.py), como sai de pre_anotar.py depois da revisão.
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

from classes import CLASSES

RAIZ = Path(__file__).resolve().parent
SAIDA_APP = RAIZ.parent / 'modelos' / 'fabrica'


def separar(pasta, dados, frac_val=0.2, semente=7):
    """Separa treino/validação por classe principal (a 1ª caixa), para toda classe ter foto nos dois lados quando possível."""
    grupos = defaultdict(list)
    for img in sorted((pasta / 'images').glob('*.jpg')):
        rot = pasta / 'labels' / (img.stem + '.txt')
        linhas = rot.read_text().split() if rot.exists() else []
        grupos[linhas[0] if linhas else 'vazia'].append(img)
    rnd = random.Random(semente)
    for parte in ('train', 'val'):
        for sub in ('images', 'labels'):
            d = dados / parte / sub
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
    for imgs in grupos.values():
        rnd.shuffle(imgs)
        n_val = max(1, round(len(imgs) * frac_val)) if len(imgs) >= 3 else 0
        for i, img in enumerate(imgs):
            parte = 'val' if i < n_val else 'train'
            shutil.copy(img, dados / parte / 'images' / img.name)
            rot = pasta / 'labels' / (img.stem + '.txt')
            if rot.exists():
                shutil.copy(rot, dados / parte / 'labels' / rot.name)
    yaml = dados / 'fabrica.yaml'
    yaml.write_text(f'path: {dados}\ntrain: train/images\nval: val/images\nnames:\n' +
                    ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pasta')
    ap.add_argument('--epocas', type=int, default=150)
    ap.add_argument('--tamanho', type=int, default=416)
    ap.add_argument('--base', default='yolo11n.pt')
    a = ap.parse_args()
    dados = RAIZ / 'dados'
    yaml = separar(Path(a.pasta), dados)
    modelo = YOLO(a.base)
    # Poucas fotos: aumento de dados forte (cor, escala, recorte, mosaico) e espelhamento horizontal. Sem espelho vertical (o piso fica embaixo).
    modelo.train(data=str(yaml), epochs=a.epocas, imgsz=a.tamanho, batch=8, device='cpu', workers=2, patience=50,
                 project=str(RAIZ / 'execucoes'), name='fabrica', exist_ok=True,
                 hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=5, translate=0.15, scale=0.6, fliplr=0.5, mosaic=1.0,
                 mixup=0.1, close_mosaic=20, plots=True)
    melhor = RAIZ / 'execucoes' / 'fabrica' / 'weights' / 'best.pt'
    metr = YOLO(str(melhor)).val(data=str(yaml), imgsz=a.tamanho, device='cpu', plots=False)
    onnx = YOLO(str(melhor)).export(format='onnx', imgsz=a.tamanho, opset=12, simplify=True, dynamic=False)
    SAIDA_APP.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, SAIDA_APP / 'yolo-fabrica.onnx')
    (SAIDA_APP / 'rotulos.json').write_text(json.dumps({'tamanho': a.tamanho, 'classes': CLASSES,
                                                        'mAP50': round(float(metr.box.map50), 3),
                                                        'mAP50_95': round(float(metr.box.map), 3)}, ensure_ascii=False, indent=1) + '\n')
    print('mAP50', metr.box.map50, 'mAP50-95', metr.box.map, '→', SAIDA_APP)


if __name__ == '__main__':
    main()
