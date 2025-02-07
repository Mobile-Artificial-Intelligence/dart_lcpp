part of '../../llama.dart';

class GrammarSamplingParams implements NativeParam {
  String str;
  String root;

  GrammarSamplingParams({
    required this.str,
    required this.root
  });

  factory GrammarSamplingParams.fromMap(Map<String, dynamic> map) => GrammarSamplingParams(
    str: map['str'],
    root: map['root']
  );

  Map<String, dynamic> toMap() => {
    'str': str,
    'root': root
  };

  @override
  grammar_sampling_params toNative() {
    final samplingParams = calloc<grammar_sampling_params>().ref;

    samplingParams.str = str.toNativeUtf8().cast<ffi.Char>();
    samplingParams.root = root.toNativeUtf8().cast<ffi.Char>();

    return samplingParams;
  }
}