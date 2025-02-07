part of '../../llama.dart';

class LlamaParams implements NativeParam {
  // path to the model file
  String modelPath;

  // only load the vocabulary, no weights
  bool? vocabOnly;

  // use mmap if possible
  bool? useMmap;

  // force system to keep model in RAM
  bool? useMlock;

  // validate model tensor data
  bool? checkTensors;

  // text context, 0 = from model
  int? nCtx;

  // logical maximum batch size that can be submitted to llama_decode
  int? nBatch;

  // physical maximum batch size
  int? nUBatch;

  // max number of sequences (i.e. distinct states for recurrent models)
  int? nSeqMax;

  // number of threads to use for generation
  int? nThreads;

  // number of threads to use for batch processing
  int? nThreadsBatch;

  // RoPE scaling type, from `enum llama_rope_scaling_type`
  RopeScalingType? ropeScalingType;

  // whether to pool (sum) embedding results by sequence id
  PoolingType? poolingType;

  // attention type to use for embeddings
  AttentionType? attentionType;

  // RoPE base frequency, 0 = from model
  double? ropeFrequencyBase;

  // RoPE frequency scaling factor, 0 = from model
  double? ropeFrequencyScale;

  // YaRN extrapolation mix factor, negative = from model
  double? yarnExtrapolationFactor;

  // YaRN magnitude scaling factor
  double? yarnAttenuationFactor;

  // YaRN low correction dim
  double? yarnBetaFast;

  // YaRN high correction dim
  double? yarnBetaSlow;

  // YaRN original context size
  int? yarnOriginalContext;

  // defragment the KV cache if holes/size > thold, < 0 disabled (default)
  double? defragmentationThreshold;

  // data type for K cache
  GgmlType? typeK;

  // data type for V cache
  GgmlType? typeV;

  // if true, extract logits for each token
  bool? logitsAll;

  // if true, extract embeddings (together with logits)
  bool? embeddings;

  // whether to offload the KQV ops (including the KV cache) to GPU
  bool? offloadKqv;

  // whether to use flash attention
  bool? flashAttention;

  // whether to measure performance timings
  bool? noPerformance;

  // whether to use greedy sampling
  bool greedy;

  // whether to use infill sampling
  bool infill;

  // seed for sampling
  int? seed;

  // top k for sampling
  int? topK;

  // top p for sampling
  PSamplingParams? topP;

  // min p for sampling
  PSamplingParams? minP;

  // typical p for sampling
  PSamplingParams? typicalP;

  // temperature for sampling
  TemperatureSamplingParams? temperature;

  // xtc for sampling
  XtcSamplingParams? xtc;

  // mirostat for sampling
  MirostatSamplingParams? mirostat;

  // mirostat v2 for sampling
  MirostatV2SamplingParams? mirostatV2;

  // grammar for sampling
  GrammarSamplingParams? grammar;

  // penalties for sampling
  PenaltiesSamplingParams? penalties;

  // dry sampling for sampling
  DrySamplingParams? dry;

  LlamaParams({
    required this.modelPath,
    this.vocabOnly,
    this.useMmap,
    this.useMlock,
    this.checkTensors,
    this.nCtx,
    this.nBatch,
    this.nUBatch,
    this.nSeqMax,
    this.nThreads,
    this.nThreadsBatch,
    this.ropeScalingType,
    this.poolingType,
    this.attentionType,
    this.ropeFrequencyBase,
    this.ropeFrequencyScale,
    this.yarnExtrapolationFactor,
    this.yarnAttenuationFactor,
    this.yarnBetaFast,
    this.yarnBetaSlow,
    this.yarnOriginalContext,
    this.defragmentationThreshold,
    this.typeK,
    this.typeV,
    this.logitsAll,
    this.embeddings,
    this.offloadKqv,
    this.flashAttention,
    this.noPerformance,
    this.greedy = false,
    this.infill = false,
    this.seed,
    this.topK,
    this.topP,
    this.minP,
    this.typicalP,
    this.temperature,
    this.xtc,
    this.mirostat,
    this.mirostatV2,
    this.grammar,
    this.penalties,
    this.dry
  });

  factory LlamaParams.fromMap(Map<String, dynamic> map) => LlamaParams(
    modelPath: map['modelPath'],
    vocabOnly: map['vocabOnly'],
    useMmap: map['useMmap'],
    useMlock: map['useMlock'],
    checkTensors: map['checkTensors'],
    nCtx: map['nCtx'],
    nBatch: map['nBatch'],
    nUBatch: map['nUBatch'],
    nSeqMax: map['nSeqMax'],
    nThreads: map['nThreads'],
    nThreadsBatch: map['nThreadsBatch'],
    ropeScalingType: RopeScalingType.values[map['ropeScalingType']],
    poolingType: PoolingType.values[map['poolingType']],
    attentionType: AttentionType.values[map['attentionType']],
    ropeFrequencyBase: map['ropeFrequencyBase'],
    ropeFrequencyScale: map['ropeFrequencyScale'],
    yarnExtrapolationFactor: map['yarnExtrapolationFactor'],
    yarnAttenuationFactor: map['yarnAttenuationFactor'],
    yarnBetaFast: map['yarnBetaFast'],
    yarnBetaSlow: map['yarnBetaSlow'],
    yarnOriginalContext: map['yarnOriginalContext'],
    defragmentationThreshold: map['defragmentationThreshold'],
    typeK: GgmlType.values[map['typeK']],
    typeV: GgmlType.values[map['typeV']],
    logitsAll: map['logitsAll'],
    embeddings: map['embeddings'],
    offloadKqv: map['offloadKqv'],
    flashAttention: map['flashAttention'],
    noPerformance: map['noPerformance'],
    greedy: map['greedy'],
    infill: map['infill'],
    seed: map['seed'],
    topK: map['topK'],
    topP: map['topP'] != null ? PSamplingParams.fromMap(map['topP']) : null,
    minP: map['minP'] != null ? PSamplingParams.fromMap(map['minP']) : null,
    typicalP: map['typicalP'] != null ? PSamplingParams.fromMap(map['typicalP']) : null,
    temperature: map['temperature'] != null ? TemperatureSamplingParams.fromMap(map['temperature']) : null,
    xtc: map['xtc'] != null ? XtcSamplingParams.fromMap(map['xtc']) : null,
    mirostat: map['mirostat'] != null ? MirostatSamplingParams.fromMap(map['mirostat']) : null,
    mirostatV2: map['mirostatV2'] != null ? MirostatV2SamplingParams.fromMap(map['mirostatV2']) : null,
    grammar: map['grammar'] != null ? GrammarSamplingParams.fromMap(map['grammar']) : null,
    penalties: map['penalties'] != null ? PenaltiesSamplingParams.fromMap(map['penalties']) : null,
    dry: map['dry'] != null ? DrySamplingParams.fromMap(map['dry']) : null
  );

  factory LlamaParams.fromJson(String source) => LlamaParams.fromMap(jsonDecode(source));

  Map<String, dynamic> toMap() => {
    'modelPath': modelPath,
    'vocabOnly': vocabOnly,
    'useMmap': useMmap,
    'useMlock': useMlock,
    'checkTensors': checkTensors,
    'nCtx': nCtx,
    'nBatch': nBatch,
    'nUBatch': nUBatch,
    'nSeqMax': nSeqMax,
    'nThreads': nThreads,
    'nThreadsBatch': nThreadsBatch,
    'ropeScalingType': ropeScalingType?.index,
    'poolingType': poolingType?.index,
    'attentionType': attentionType?.index,
    'ropeFrequencyBase': ropeFrequencyBase,
    'ropeFrequencyScale': ropeFrequencyScale,
    'yarnExtrapolationFactor': yarnExtrapolationFactor,
    'yarnAttenuationFactor': yarnAttenuationFactor,
    'yarnBetaFast': yarnBetaFast,
    'yarnBetaSlow': yarnBetaSlow,
    'yarnOriginalContext': yarnOriginalContext,
    'defragmentationThreshold': defragmentationThreshold,
    'typeK': typeK?.index,
    'typeV': typeV?.index,
    'logitsAll': logitsAll,
    'embeddings': embeddings,
    'offloadKqv': offloadKqv,
    'flashAttention': flashAttention,
    'noPerformance': noPerformance,
    'greedy': greedy,
    'infill': infill,
    'seed': seed,
    'topK': topK,
    'topP': topP?.toMap(),
    'minP': minP?.toMap(),
    'typicalP': typicalP?.toMap(),
    'temperature': temperature?.toMap(),
    'xtc': xtc?.toMap(),
    'mirostat': mirostat?.toMap(),
    'mirostatV2': mirostatV2?.toMap(),
    'grammar': grammar?.toMap(),
    'penalties': penalties?.toMap(),
    'dry': dry?.toMap()
  };

  String toJson() => jsonEncode(toMap());

  @override
  ffi.Pointer<api_params> toNative() {
    final params = lib.api_default_params();

    params.ref.model_path = modelPath.toNativeUtf8().cast<ffi.Char>();
    
    if (vocabOnly != null) {
      params.ref.vocab_only = vocabOnly!;
    }

    if (useMmap != null) {
      params.ref.use_mmap = useMmap!;
    }

    if (useMlock != null) {
      params.ref.use_mlock = useMlock!;
    }

    if (checkTensors != null) {
      params.ref.check_tensors = checkTensors!;
    }

    if (nCtx != null) {
      params.ref.n_ctx = nCtx!;
    }

    if (nBatch != null) {
      params.ref.n_batch = nBatch!;
    }

    if (nUBatch != null) {
      params.ref.n_ubatch = nUBatch!;
    }

    if (nSeqMax != null) {
      params.ref.n_seq_max = nSeqMax!;
    }

    if (nThreads != null) {
      params.ref.n_threads = nThreads!;
    }

    if (nThreadsBatch != null) {
      params.ref.n_threads_batch = nThreadsBatch!;
    }

    if (ropeScalingType != null) {
      params.ref.rope_scaling_type = ropeScalingType!.index;
    }

    if (poolingType != null) {
      params.ref.pooling_type = poolingType!.index;
    }

    if (attentionType != null) {
      params.ref.attention_type = attentionType!.index;
    }

    if (ropeFrequencyBase != null) {
      params.ref.rope_freq_base = ropeFrequencyBase!;
    }

    if (ropeFrequencyScale != null) {
      params.ref.rope_freq_scale = ropeFrequencyScale!;
    }

    if (yarnExtrapolationFactor != null) {
      params.ref.yarn_ext_factor = yarnExtrapolationFactor!;
    }

    if (yarnAttenuationFactor != null) {
      params.ref.yarn_attn_factor = yarnAttenuationFactor!;
    }

    if (yarnBetaFast != null) {
      params.ref.yarn_beta_fast = yarnBetaFast!;
    }

    if (yarnBetaSlow != null) {
      params.ref.yarn_beta_slow = yarnBetaSlow!;
    }

    if (yarnOriginalContext != null) {
      params.ref.yarn_orig_ctx = yarnOriginalContext!;
    }

    if (defragmentationThreshold != null) {
      params.ref.defrag_thold = defragmentationThreshold!;
    }

    if (typeK != null) {
      params.ref.type_k = typeK!.index;
    }

    if (typeV != null) {
      params.ref.type_v = typeV!.index;
    }

    if (logitsAll != null) {
      params.ref.logits_all = logitsAll!;
    }

    if (embeddings != null) {
      params.ref.embeddings = embeddings!;
    }

    if (offloadKqv != null) {
      params.ref.offload_kqv = offloadKqv!;
    }

    if (flashAttention != null) {
      params.ref.flash_attn = flashAttention!;
    }
    
    if (noPerformance != null) {
      params.ref.no_perf = noPerformance!;
    }

    params.ref.greedy = greedy;
    params.ref.infill = infill;

    if (seed != null) {
      params.ref.seed = seed!;
    }

    if (topK != null) {
      params.ref.top_k = topK!;
    }

    if (topP != null) {
      params.ref.top_p = topP!.toNative();
    }

    if (minP != null) {
      params.ref.min_p = minP!.toNative();
    }

    if (typicalP != null) {
      params.ref.typical_p = typicalP!.toNative();
    }

    if (temperature != null) {
      params.ref.temperature = temperature!.toNative();
    }

    if (xtc != null) {
      params.ref.xtc = xtc!.toNative();
    }

    if (mirostat != null) {
      params.ref.mirostat = mirostat!.toNative();
    }

    if (mirostatV2 != null) {
      params.ref.mirostat_v2 = mirostatV2!.toNative();
    }

    if (grammar != null) {
      params.ref.grammar = grammar!.toNative();
    }

    if (penalties != null) {
      params.ref.penalties = penalties!.toNative();
    }

    if (dry != null) {
      params.ref.dry = dry!.toNative();
    }

    return params;
  }
}