part of '../../llama.dart';

class PSamplingParams implements NativeParam {
  double p;
  int minKeep;

  PSamplingParams({
    required this.p,
    required this.minKeep
  });

  factory PSamplingParams.fromMap(Map<String, dynamic> map) => PSamplingParams(
    p: map['p'],
    minKeep: map['minKeep']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    'minKeep': minKeep
  };

  @override
  p_sampling_params toNative() {
    final samplingParams = calloc<p_sampling_params>().ref;

    samplingParams.p = p;
    samplingParams.min_keep = minKeep;

    return samplingParams;
  }
}