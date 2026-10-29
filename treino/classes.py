# Classes do modelo da fábrica. A ORDEM define o índice no YOLO e precisa bater com js/nucleo.js (CLASSES).
# prefixos: início do nome do arquivo da foto → classe principal da foto
# prompts: descrição em inglês usada pelo YOLOE na pré-anotação
CLASSES = ['caixa', 'carteira', 'escada', 'lixeira', 'manequim', 'mesa', 'banco', 'planta']
PROMPTS = {
    'caixa': 'cardboard box', 'carteira': 'school chair with writing tablet', 'escada': 'step ladder',
    'lixeira': 'trash can', 'manequim': 'mannequin', 'mesa': 'table', 'banco': 'bench', 'planta': 'potted plant',
}
PREFIXOS = {'caixa': 'caixa', 'carteira': 'carteira', 'escada': 'escada', 'lixo': 'lixeira', 'manequim': 'manequim',
            'multiplos_manequim': 'manequim', 'mesa': 'mesa', 'banco': 'banco', 'planta': 'planta'}


def classe_do_arquivo(nome):
    """Classe principal pela convenção de nomes do banco de fotos (ex.: caixa_frente.jpg, multiplos_manequim_lado_1.jpg)."""
    base = nome.lower()
    for p in sorted(PREFIXOS, key=len, reverse=True):
        if base.startswith(p):
            return PREFIXOS[p]
    return None
