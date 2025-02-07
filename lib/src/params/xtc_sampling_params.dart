part of '../../llama.dart';

class XtcSamplingParams implements NativeParam {
  double p;
  double t;
  int minKeep;
  int seed;

  XtcSamplingParams({
    required this.p,
    required this.t,
    required this.minKeep,
    required this.seed
  });

  factory XtcSamplingParams.fromMap(Map<String, dynamic> map) => XtcSamplingParams(
    p: map['p'],
    t: map['t'],
    minKeep: map['minKeep'],
    seed: map['seed']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    't': t,
    'minKeep': minKeep,
    'seed': seed
  };

  @override
  ffi.Pointer<xtc_sampling_params> toNative() {
    final samplingParams = calloc<xtc_sampling_params>();

    samplingParams.ref.probability = p;
    samplingParams.ref.threshold = t;
    samplingParams.ref.min_keep = minKeep;
    samplingParams.ref.seed = seed;

    return samplingParams;
  }
}