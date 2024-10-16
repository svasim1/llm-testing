import torch
import transformers
import logging

logging.basicConfig(level=logging.DEBUG)

model_id = "AI-Sweden-Models/Llama-3-8B-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

context = '# kontext:\nEXEMPEL (DE FÖRSTA 5 av 100):\nFortkörning förekommer ofta på gatan.\n\n' \
          'Fullständigt kaos med bilar parkerade på trottoarer, i vägkorsningar och på privat mark. ' \
          'Vansinneskörningar i hög hastighit och allmänt ignorerande av trafikregler (t.ex. väjningsplikt) ' \
          'med stor fara för oskyddade trafikanter. Problem är återkommande.\n\n' \
          'Bilister i Tygelsjö kör alldeles för fort! Hastighetsgränsen på 40 respekteras inte [nr] gränsen' \
          ' framför skolarna respekteras ännu mindre. Man borde sätta upp digitala hastighetsskyltar som det ' \
          'finns i exempelvis Bunkeflostrand och Käglinge.\n\nfrågor om fortköning\n\nPå landmannagatan och ' \
          'Kopparbergsgatan körs det för snabbt och med för högljudda fordon. Detta händer åtminstone 20 ' \
          'gånger varje kväll. På Lantmannagatan där det finns farthinder är dint lika illa som på ' \
          'Kopparbergsgatan där det inte finns några hinder. trafiken gör att det är svårt at tha ' \
          'fönster öppna på kvällar, i synnerhet jag som har en liten bebis som vaknar av de kraftiga' \
          ' och plötsliga ljuden från dessa fortkörare. Det är alltså inte den normala trafiken som är problemet, ' \
          'utan de alltför många som kö snabbt och med starka ljud från bilen.'

question = "På vilka platser är fortkörning ett problem?"

messages = [
    {"role": "system", "content": "Du är en svensktalande assistent som sammanfattar text."},
    {"role": "user",
     "content": "Svara på frågan genom att sammanfatta innehållet i exemplen med egna ord. "
                f"Håll sammanfattningen kort och koncis.\n # Fråga {question}\n # Exempel:\n{context}"
     },
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
    eos_token_id=terminators,
    pad_token_id=pipeline.tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])