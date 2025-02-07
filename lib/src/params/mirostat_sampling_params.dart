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
  mirostat_sampling_params toNative() {
    final samplingParams = calloc<mirostat_sampling_params>().ref;

    samplingParams.n_vocab = nVocab;
    samplingParams.seed = seed;
    samplingParams.tau = tau;
    samplingParams.eta = eta;
    samplingParams.m = m;

    return samplingParams;
  }
}