# Treino do modelo da fábrica (YOLO nano)

Modelo próprio para o ambiente controlado, no lugar do COCO-SSD. Roda no navegador (ONNX Runtime Web, WebAssembly), escolhido no app em **Modelo → Fábrica**.
Nada de Teachable Machine: é um YOLO11n treinado com o Ultralytics e exportado para ONNX.

Classes (a ordem importa e precisa bater com `js/nucleo.js`): `caixa, carteira, escada, lixeira, manequim, mesa, banco, planta` (ver `classes.py`).

## Passo a passo

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r treino/requirements.txt

# 1. (opcional) vídeos → fotos: grave andando em volta de cada objeto, nome do vídeo = classe (caixa_1.mp4, manequim_2.mp4...)
python treino/extrair_quadros.py videos/ fotos/ --por-segundo 2

# 2. pré-anotação automática (YOLOE com prompts de texto): gera images/, labels/, revisao/ e classes.txt
python treino/pre_anotar.py fotos/ anotado/

# 3. REVISE as caixas. Olhe anotado/revisao/*.jpg; para corrigir, abra https://www.makesense.ai,
#    carregue anotado/images, importe anotado/labels + classes.txt (formato YOLO), ajuste e exporte de volta para anotado/labels.

# 4. treino + exportação: grava modelos/fabrica/yolo-fabrica.onnx e modelos/fabrica/rotulos.json
python treino/treinar.py anotado/ --epocas 150 --tamanho 416
```

Sem placa de vídeo, o treino roda na CPU (o piloto de 33 fotos levou cerca de 20 minutos em 4 núcleos). Com mais fotos, o Google Colab com GPU é bem mais rápido: os mesmos comandos funcionam lá.

## Regras para as fotos

- **40 a 80 fotos por classe**, no mínimo. Distâncias variadas (perto, 2 m, 5 m), ângulos e alturas da câmera do celular como o usuário vai segurar.
- Luz diferente (manhã, tarde, luz acesa/apagada), objeto cortado na borda, mais de um objeto na cena.
- **Fotos sem nenhum objeto** (corredor vazio, parede, porta): ensinam o modelo a não inventar obstáculos. Basta deixar o .txt vazio.
- **Toda** ocorrência das classes na foto precisa de caixa, inclusive no fundo. Objeto sem caixa ensina o modelo a ignorá-lo.
- A carteira inclui a prancheta; o manequim inclui o pedestal (é o que bate na pessoa).

## Limites

- O modelo da fábrica **não** detecta pessoas nem veículos (não estavam no treino). Para isso, continue usando o COCO-SSD ou inclua essas classes nas fotos.
- O piloto foi treinado com só 33 fotos: serve para provar o caminho de ponta a ponta, não para uso real. A métrica fica em `modelos/fabrica/rotulos.json`.
