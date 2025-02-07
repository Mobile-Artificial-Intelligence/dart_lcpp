part of '../../llama.dart';

class TemperatureSamplingParams implements NativeParam {
  double temperature;
  double? delta;
  double? exponent;

  TemperatureSamplingParams({
    required this.temperature,
    this.delta,
    this.exponent
  });

  factory TemperatureSamplingParams.fromMap(Map<String, dynamic> map) => TemperatureSamplingParams(
    temperature: map['temperature'],
    delta: map['delta'],
    exponent: map['exponent']
  );

  Map<String, dynamic> toMap() => {
    'temperature': temperature,
    'delta': delta,
    'exponent': exponent
  };

  @override
  ffi.Pointer<temperature_sampling_params> toNative() {
    final samplingParams = calloc<temperature_sampling_params>();

    samplingParams.ref.temperature = temperature;
    samplingParams.ref.delta = delta ?? -1;
    samplingParams.ref.exponent = exponent ?? -1;

    return samplingParams;
  }
}