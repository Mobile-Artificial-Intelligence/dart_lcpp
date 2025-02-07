part of '../../llama.dart';

class PenaltiesSamplingParams implements NativeParam {
  int lastN;
  double repeat;
  double frequency;
  double present;

  PenaltiesSamplingParams({
    required this.lastN,
    required this.repeat,
    required this.frequency,
    required this.present
  });

  factory PenaltiesSamplingParams.fromMap(Map<String, dynamic> map) => PenaltiesSamplingParams(
    lastN: map['lastN'],
    repeat: map['repeat'],
    frequency: map['frequency'],
    present: map['present']
  );

  Map<String, dynamic> toMap() => {
    'lastN': lastN,
    'repeat': repeat,
    'frequency': frequency,
    'present': present
  };

  @override
  penalties_sampling_params toNative() {
    final samplingParams = calloc<penalties_sampling_params>().ref;

    samplingParams.last_n = lastN;
    samplingParams.repeat = repeat;
    samplingParams.freq = frequency;
    samplingParams.present = present;

    return samplingParams;
  }
}