import os, time
import onnxruntime_genai as og

model = og.Model(os.environ["MODEL_DIR"])
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()

prompt = "<|user|>\nExplica en dos frases qué es un JSON Schema.<|end|>\n<|assistant|>\n"
entrada = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=len(entrada) + 200)

generator = og.Generator(model, params)
generator.append_tokens(entrada)

inicio, n = time.time(), 0
while not generator.is_done():
    generator.generate_next_token()
    print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
    n += 1

print(f"\n\n{n} tokens, {n / (time.time() - inicio):.1f} tokens/s")
