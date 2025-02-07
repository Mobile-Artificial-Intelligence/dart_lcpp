part of '../../llama.dart';

class DrySamplingParams implements NativeParam {
  int nCtxTrain;
  double multiplier;
  double dryBase;
  int allowedLength;
  int penaltyLastN;
  List<String> sequenceBreakers;

  DrySamplingParams({
    required this.nCtxTrain,
    required this.multiplier,
    required this.dryBase,
    required this.allowedLength,
    required this.penaltyLastN,
    required this.sequenceBreakers
  });

  factory DrySamplingParams.fromMap(Map<String, dynamic> map) => DrySamplingParams(
    nCtxTrain: map['nCtxTrain'],
    multiplier: map['multiplier'],
    dryBase: map['dryBase'],
    allowedLength: map['allowedLength'],
    penaltyLastN: map['penaltyLastN'],
    sequenceBreakers: List<String>.from(map['sequenceBreakers'])
  );

  Map<String, dynamic> toMap() => {
    'nCtxTrain': nCtxTrain,
    'multiplier': multiplier,
    'dryBase': dryBase,
    'allowedLength': allowedLength,
    'penaltyLastN': penaltyLastN,
    'sequenceBreakers': sequenceBreakers
  };

  @override
  dry_sampling_params toNative() {
    final drySamplingParams = calloc<dry_sampling_params>().ref;

    drySamplingParams.n_ctx_train = nCtxTrain;
    drySamplingParams.multiplier = multiplier;
    drySamplingParams.base = dryBase;
    drySamplingParams.allowed_length = allowedLength;
    drySamplingParams.penalty_last_n = penaltyLastN;
    drySamplingParams.breakers = convertListToPointer(sequenceBreakers);
    drySamplingParams.num_breakers = sequenceBreakers.length;

    return drySamplingParams;
  }
}