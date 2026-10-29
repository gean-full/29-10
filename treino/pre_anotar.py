"""Pré-anotação automática: desenha as caixas com o YOLOE (vocabulário aberto, prompts de texto) e grava rótulos no formato YOLO.

Uso:  python pre_anotar.py PASTA_FOTOS PASTA_SAIDA
Saída: PASTA_SAIDA/images/*.jpg, PASTA_SAIDA/labels/*.txt (classe cx cy w h, normalizados), PASTA_SAIDA/revisao/*.jpg (caixas desenhadas)
       e PASTA_SAIDA/classes.txt (para importar no makesense.ai, CVAT ou Label Studio e corrigir à mão).
As caixas são um RASCUNHO: revise todas antes de treinar.
"""
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLOE

from classes import CLASSES, PROMPTS, classe_do_arquivo

CONF = 0.08          # baixo de propósito: melhor sobrar caixa para apagar do que faltar
CONF_OUTRAS = 0.25   # objetos de outra classe que aparecem na foto precisam de mais confiança


def main(origem, destino):
    origem, destino = Path(origem), Path(destino)
    for sub in ('images', 'labels', 'revisao'):
        (destino / sub).mkdir(parents=True, exist_ok=True)
    (destino / 'classes.txt').write_text('\n'.join(CLASSES) + '\n')
    modelo = YOLOE('yoloe-11l-seg.pt')   # pesos e codificador de texto (MobileCLIP) vêm dos assets do Ultralytics no GitHub
    textos = [PROMPTS[c] for c in CLASSES]
    modelo.set_classes(textos, modelo.get_text_pe(textos))
    resumo = []
    for foto in sorted(p for p in origem.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')):
        img = ImageOps.exif_transpose(Image.open(foto)).convert('RGB')   # aplica a rotação do celular
        img.save(destino / 'images' / (foto.stem + '.jpg'), quality=95)
        W, H = img.size
        principal = classe_do_arquivo(foto.name)
        r = modelo.predict(img, conf=CONF, iou=0.5, verbose=False)[0]
        caixas = []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            nome = CLASSES[int(c)]
            if nome != principal and s < CONF_OUTRAS:
                continue
            caixas.append((nome, s, b))
        # a classe do nome do arquivo tem que aparecer: se o detector não achou, fica a melhor caixa de qualquer classe reetiquetada
        if principal and not any(n == principal for n, _, _ in caixas) and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            caixas.append((principal, float(r.boxes.conf[i]), r.boxes.xyxy[i].tolist()))
        linhas, desenho = [], ImageDraw.Draw(img)
        for nome, s, (x1, y1, x2, y2) in caixas:
            linhas.append(f'{CLASSES.index(nome)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
            desenho.rectangle([x1, y1, x2, y2], outline='red', width=max(3, W // 150))
            desenho.text((x1 + 6, y1 + 6), f'{nome} {s:.2f}', fill='red')
        (destino / 'labels' / (foto.stem + '.txt')).write_text('\n'.join(linhas) + ('\n' if linhas else ''))
        img.save(destino / 'revisao' / (foto.stem + '.jpg'), quality=80)
        resumo.append((foto.name, principal, [(n, round(s, 2)) for n, s, _ in caixas]))
    for linha in resumo:
        print(*linha)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
