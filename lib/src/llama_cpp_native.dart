part of '../llama.dart';

class LlamaCppNative {
  final ffi.Pointer<llama_model> _model;
  final ffi.Pointer<llama_context> _context;
  final ffi.Pointer<llama_sampler> _sampler;

  Completer? _completer;

  LlamaCppNative({
    required ffi.Pointer<llama_model> model, 
    required ffi.Pointer<llama_context> context, 
    required ffi.Pointer<llama_sampler> sampler
  }) : _model = model, _context = context, _sampler = sampler;

  factory LlamaCppNative.fromParams(
    String modelPath, 
    ModelParams modelParams, 
    ContextParams contextParams, 
    SamplingParams samplingParams
  ) {
    lib.ggml_backend_load_all();
    lib.llama_backend_init();
    log("backend loaded");

    final nativeModelParams = modelParams.toNative();
    final nativeModelPath = modelPath.toNativeUtf8().cast<ffi.Char>();
      
    final model = lib.llama_load_model_from_file(
      nativeModelPath, 
      nativeModelParams
    );
    assert(model.address != 0, 'Failed to load model');

    malloc.free(nativeModelPath);
    log("Model loaded");

    final nativeContextParams = contextParams.toNative();

    final context = lib.llama_init_from_model(model, nativeContextParams);
    assert(context.address != 0, 'Failed to initialize context');
    log("Context initialized");

    final vocab = lib.llama_model_get_vocab(model);
    final sampler = samplingParams.toNative(vocab);
    assert(sampler.address != 0, 'Failed to initialize sampler');
    log("Sampler initialized");

    return LlamaCppNative(model: model, context: context, sampler: sampler);
  }

  Stream<String> prompt(List<ChatMessage> messages) {
    assert(_model.address != 0, 'Model is not loaded');
    assert(_context.address != 0, 'Context is not initialized');
    assert(_sampler.address != 0, 'Sampler is not initialized');

    _completer = Completer();

    final nCtx = lib.llama_n_ctx(_context);

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = lib.llama_model_chat_template(_model, ffi.nullptr);

    int contextLength = lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      true, 
      formatted, 
      nCtx
    );

    if (contextLength > nCtx) {
      formatted = calloc<ffi.Char>(contextLength);
      contextLength = lib.llama_chat_apply_template(
        template, 
        messages.toNative(), 
        messages.length, 
        true, 
        formatted, 
        contextLength
      );
    }

    if (contextLength < 0) {
      throw Exception('Failed to apply template');
    }

    final prompt = formatted.cast<Utf8>().toDartString();

    return _generate(prompt);
  }

  Stream<String> _generate(String prompt) async* {
    final vocab = lib.llama_model_get_vocab(_model);
    final isFirst = lib.llama_get_kv_cache_used_cells(_context) == 0;

    final nPromptTokens = -lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw Exception('Failed to tokenize');
    }

    llama_batch batch = lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;
    
    while (!_completer!.isCompleted) {
      final nCtx = lib.llama_n_ctx(_context);
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(_context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw Exception('Context size exceeded');
      }

      if (lib.llama_decode(_context, batch) != 0) {
        throw Exception('Failed to decode');
      }

      newTokenId = lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      if (lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true) < 0) {
        throw Exception('Failed to convert token to piece');
      }

      try {
        yield buffer.cast<Utf8>().toDartString();
      }
      catch (e) {
        throw FormatException('Failed to parse UTF-8 string from buffer', e);
      }
      finally {
        calloc.free(buffer);
      }

      final newTokenPointer = calloc<llama_token>(1);
      newTokenPointer.value = newTokenId;

      batch = lib.llama_batch_get_one(newTokenPointer, 1);
      calloc.free(newTokenPointer);
    }

    lib.llama_batch_free(batch);
    calloc.free(promptTokens);
  }

  void stop() {
    _completer?.complete();
  }

  void free() {
    lib.llama_sampler_free(_sampler);
    lib.llama_free(_context);
    lib.llama_free_model(_model);
  }
}