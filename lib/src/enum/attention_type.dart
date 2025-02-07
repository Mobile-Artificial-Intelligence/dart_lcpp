part of '../../llama.dart';

enum AttentionType {
  unspecified,
  causal,
  nonCausal;

  static AttentionType fromString(String value) {
    switch (value) {
      case 'causal':
        return AttentionType.causal;
      case 'non-causal':
        return AttentionType.nonCausal;
      default:
        return AttentionType.unspecified;
    }
  }
}