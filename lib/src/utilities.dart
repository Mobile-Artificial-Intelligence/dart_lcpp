part of '../llama.dart';

ffi.Pointer<ffi.Pointer<ffi.Char>> convertListToPointer(List<String> strings) {
  // Allocate memory for the array of pointers
  final ffi.Pointer<ffi.Pointer<ffi.Char>> pointerArray = malloc.allocate(ffi.sizeOf<ffi.Pointer<ffi.Char>>() * strings.length);

  for (int i = 0; i < strings.length; i++) {
    // Convert each string to Utf8 and store its pointer
    pointerArray[i] = strings[i].toNativeUtf8().cast<ffi.Char>();
  }

  return pointerArray;
}