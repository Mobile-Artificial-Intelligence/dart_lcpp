part of '../llama.dart';

class LlamaNative {
  static StreamController<String> _controller = StreamController<String>();

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

  Stream<String> prompt(List<ChatMessage> messages) async* {
    _asyncPrompt(messages);

    yield* _controller.stream;
  }

  Future<int> _asyncPrompt(List<ChatMessage> messages) async {
    final msg = messages.toNative();

    return lib.api_prompt(msg, messages.length, ffi.Pointer.fromFunction(_output));
  }

  static void _output(ffi.Pointer<ffi.Char> buffer) {
    _controller.add(buffer.cast<Utf8>().toDartString());
  }
}