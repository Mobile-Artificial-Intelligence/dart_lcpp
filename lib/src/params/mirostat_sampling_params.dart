part of '../../llama.dart';

class MirostatSamplingParams implements NativeParam {
  int nVocab;
  int seed;
  double tau;
  double eta;
  int m;

  MirostatSamplingParams({
    required this.nVocab,
    required this.seed,
    required this.tau,
    required this.eta,
    required this.m
  });

  factory MirostatSamplingParams.fromMap(Map<String, dynamic> map) => MirostatSamplingParams(
    nVocab: map['nVocab'],
    seed: map['seed'],
    tau: map['tau'],
    eta: map['eta'],
    m: map['m']
  );

  Map<String, dynamic> toMap() => {
    'nVocab': nVocab,
    'seed': seed,
    'tau': tau,
    'eta': eta,
    'm': m
  };

  @override
  ffi.Pointer<mirostat_sampling_params> toNative() {
    final samplingParams = calloc<mirostat_sampling_params>();

    samplingParams.ref.n_vocab = nVocab;
    samplingParams.ref.seed = seed;
    samplingParams.ref.tau = tau;
    samplingParams.ref.eta = eta;
    samplingParams.ref.m = m;

    return samplingParams;
  }
}