part of '../../llama.dart';

enum PoolingType {
  unspecified,
  none,
  mean,
  cls,
  last,
  rank;

  static PoolingType fromString(String value) {
    switch (value) {
      case 'none':
        return PoolingType.none;
      case 'mean':
        return PoolingType.mean;
      case 'cls':
        return PoolingType.cls;
      case 'last':
        return PoolingType.last;
      case 'rank':
        return PoolingType.rank;
      default:
        return PoolingType.unspecified;
    }
  }
}