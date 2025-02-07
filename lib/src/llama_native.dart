part of '../llama.dart';

class LlamaNative {
  LlamaNative(
    api_params params, 
  ) {
    if (lib.api_init(params) != 0) {
      throw Exception('Failed to initialize API');
    }
  }

  factory LlamaNative.fromParams(
    LlamaParams params
  ) => LlamaNative(params.toNative());
}