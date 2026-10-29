'use strict';
/*
 * Detector próprio (YOLO nano treinado com as fotos do ambiente) rodando no navegador com ONNX Runtime Web (WebAssembly).
 * Expõe a MESMA interface do COCO-SSD, para o resto do app não mudar:
 *   VisioYolo.carregar({modelo, rotulos}) → Promise<{detect(fonte, maxDet, scoreMin) → [{class, score, bbox:[x,y,w,h]}], dispose()}>
 * - modelo: URL do .onnx exportado pelo Ultralytics (saída [1, 4 + nClasses, N], caixas em cx,cy,w,h no espaço da entrada);
 * - rotulos: URL do JSON {"tamanho":416, "classes":["caixa", ...]} gerado junto com o modelo (ordem = índice da classe).
 * A biblioteca ONNX Runtime só é baixada quando este modelo é escolhido; nenhuma imagem sai do aparelho.
 */
(function(g){
  const ORT_VERSAO = '1.20.1';
  const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSAO + '/dist/';
  const ORT_SRI = 'sha384-6hgW9I+eQKl5CrHLstudPd4L3E2eTYHm+qNKvf4pI+RZnfBJILOHAiaalHV+trgG';
  let ortPromessa = null;

  function carregarOrt(){
    if (g.ort) return Promise.resolve(g.ort);
    if (ortPromessa) return ortPromessa;
    ortPromessa = new Promise((ok, falha) => {
      const s = document.createElement('script');
      s.src = ORT_BASE + 'ort.wasm.min.js'; s.integrity = ORT_SRI; s.crossOrigin = 'anonymous';
      s.onload = () => g.ort ? ok(g.ort) : falha(new Error('ort ausente'));
      s.onerror = () => { ortPromessa = null; falha(Object.assign(new Error('ort não carregou'),
        {mensagemUsuario:'A biblioteca do modelo da fábrica não foi carregada (sem internet ou bloqueada pela rede). Recarregue a página com internet.'})); };
      document.head.appendChild(s);
    });
    return ortPromessa;
  }

  // IoU entre caixas [x1,y1,x2,y2]
  function iou(a, b){
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (w <= 0 || h <= 0) return 0;
    const i = w*h; return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i);
  }

  // Decodifica a saída do YOLO (formato [4 + nc, N], transposto) e aplica NMS por classe. Puro: testável no Node.
  function decodificar(dados, nc, n, o){
    const cand = [];
    for (let j = 0; j < n; j++){
      let melhor = -1, sc = o.scoreMin;
      for (let c = 0; c < nc; c++){ const v = dados[(4+c)*n + j]; if (v > sc){ sc = v; melhor = c; } }
      if (melhor < 0) continue;
      const cx = dados[j], cy = dados[n+j], w = dados[2*n+j], h = dados[3*n+j];
      // desfaz o letterbox: espaço da entrada → pixels da fonte
      const x1 = (cx - w/2 - o.padX) / o.escala, y1 = (cy - h/2 - o.padY) / o.escala;
      const x2 = (cx + w/2 - o.padX) / o.escala, y2 = (cy + h/2 - o.padY) / o.escala;
      cand.push({c:melhor, s:sc, b:[Math.max(0, x1), Math.max(0, y1), Math.min(o.W, x2), Math.min(o.H, y2)]});
    }
    cand.sort((a, b) => b.s - a.s);
    const fica = [];
    for (const k of cand){
      if (fica.length >= o.maxDet) break;
      if (!fica.some(f => f.c === k.c && iou(f.b, k.b) > (o.iouMax ?? .45))) fica.push(k);
    }
    return fica.map(k => ({class:o.classes[k.c], score:k.s, bbox:[k.b[0], k.b[1], k.b[2]-k.b[0], k.b[3]-k.b[1]]}));
  }

  async function carregar(op){
    const [ort, meta] = await Promise.all([carregarOrt(), fetch(op.rotulos).then(r => { if (!r.ok) throw new Error('rótulos ' + r.status); return r.json(); })]);
    const T = meta.tamanho, classes = meta.classes;
    if (!Number.isInteger(T) || !Array.isArray(classes) || !classes.length) throw new Error('rótulos do modelo inválidos');
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = g.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;   // threads exigem isolamento de origem
    const sessao = await ort.InferenceSession.create(op.modelo, {executionProviders:['wasm'], graphOptimizationLevel:'all'});
    const tela = document.createElement('canvas'); tela.width = T; tela.height = T;
    const ctx = tela.getContext('2d', {willReadFrequently:true});
    const entrada = new Float32Array(3*T*T), plano = T*T;
    let livre = false;
    return {
      classes, tamanho: T,
      async detect(fonte, maxDet = 20, scoreMin = .3){
        if (livre) throw new Error('modelo liberado');
        const W = fonte.videoWidth || fonte.naturalWidth || fonte.width, H = fonte.videoHeight || fonte.naturalHeight || fonte.height;
        if (!W || !H) return [];
        const escala = Math.min(T/W, T/H), w = Math.round(W*escala), h = Math.round(H*escala), padX = (T-w)/2, padY = (T-h)/2;
        ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0, 0, T, T);
        ctx.drawImage(fonte, padX, padY, w, h);
        const px = ctx.getImageData(0, 0, T, T).data;
        for (let i = 0; i < plano; i++){ entrada[i] = px[4*i]/255; entrada[plano+i] = px[4*i+1]/255; entrada[2*plano+i] = px[4*i+2]/255; }
        const saida = await sessao.run({[sessao.inputNames[0]]: new ort.Tensor('float32', entrada, [1, 3, T, T])});
        const t = saida[sessao.outputNames[0]], nc = t.dims[1] - 4, n = t.dims[2];
        return decodificar(t.data, nc, n, {scoreMin, maxDet, padX, padY, escala, W, H, classes});
      },
      dispose(){ livre = true; try { sessao.release && sessao.release(); } catch (e) {} }
    };
  }

  const api = {carregar, decodificar, iou};
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else g.VisioYolo = api;
})(typeof window !== 'undefined' ? window : globalThis);
