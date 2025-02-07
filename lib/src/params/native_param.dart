part of '../../llama.dart';

abstract class NativeParam {
  ffi.Pointer<ffi.NativeType> toNative();
}