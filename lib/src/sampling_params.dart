part of '../llama.dart';



class SamplingParams {

  TemperatureArguments? temperature;
  XtcArguments? xtc;
  MirostatArguments? mirostat;
  MirostatV2Arguments? mirostatV2;
  GrammarArguments? grammar;
  PenaltiesArguments? penalties;
  DrySamplerArguments? drySampler;

  SamplingParams({
    this.temperature,
    this.xtc,
    this.mirostat,
    this.mirostatV2,
    this.grammar,
    this.penalties,
    this.drySampler
  });

  factory SamplingParams.fromMap(Map<String, dynamic> map) => SamplingParams(
    greedy: map['greedy'],
    infill: map['infill'],
    seed: map['seed'],
    topK: map['topK'],
    topP: map['topP'] != null ? TopPArguments.fromMap(map['topP']) : null,
    minP: map['minP'] != null ? MinPArguments.fromMap(map['minP']) : null,
    typicalP: map['typicalP'] != null ? TypicalPArguments.fromMap(map['typicalP']) : null,
    temperature: map['temperature'] != null ? TemperatureArguments.fromMap(map['temperature']) : null,
    xtc: map['xtc'] != null ? XtcArguments.fromMap(map['xtc']) : null,
    mirostat: map['mirostat'] != null ? MirostatArguments.fromMap(map['mirostat']) : null,
    mirostatV2: map['mirostatV2'] != null ? MirostatV2Arguments.fromMap(map['mirostatV2']) : null,
    grammar: map['grammar'] != null ? GrammarArguments.fromMap(map['grammar']) : null,
    penalties: map['penalties'] != null ? PenaltiesArguments.fromMap(map['penalties']) : null,
    drySampler: map['drySampler'] != null ? DrySamplerArguments.fromMap(map['drySampler']) : null
  );

  factory SamplingParams.fromJson(String source) => SamplingParams.fromMap(jsonDecode(source));

  ffi.Pointer<llama_sampler> toNative(ffi.Pointer<llama_vocab> vocab) {
    final sampler = lib.llama_sampler_chain_init(lib.llama_sampler_chain_default_params());

    if (greedy) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_greedy());
    }

    if (infill) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_infill(vocab));
    }

    if (seed != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(seed!));
    }

    if (topK != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(topK!));
    }

    topP?.add(sampler);

    minP?.add(sampler);

    typicalP?.add(sampler);

    temperature?.add(sampler);

    xtc?.add(sampler);

    mirostat?.add(sampler);

    mirostatV2?.add(sampler);

    grammar?.add(sampler, vocab);

    penalties?.add(sampler);

    drySampler?.add(sampler, vocab);

    return sampler;
  }

  Map<String, dynamic> toMap() => {
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
    'drySampler': drySampler?.toMap()
  };

  String toJson() => jsonEncode(toMap());
}