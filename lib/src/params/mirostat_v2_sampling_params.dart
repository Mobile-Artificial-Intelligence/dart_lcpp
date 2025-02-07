part of '../../llama.dart';

class MirostatV2SamplingParams implements NativeParam {
  int seed;
  double tau;
  double eta;

  MirostatV2SamplingParams({
    required this.seed,
    required this.tau,
    required this.eta
  });

  factory MirostatV2SamplingParams.fromMap(Map<String, dynamic> map) => MirostatV2SamplingParams(
    seed: map['seed'],
    tau: map['tau'],
    eta: map['eta']
  );

  Map<String, dynamic> toMap() => {
    'seed': seed,
    'tau': tau,
    'eta': eta
  };

  @override
  mirostat_v2_sampling_params toNative() {
    final samplingParams = calloc<mirostat_v2_sampling_params>().ref;

    samplingParams.seed = seed;
    samplingParams.tau = tau;
    samplingParams.eta = eta;

    return samplingParams;
  }
}