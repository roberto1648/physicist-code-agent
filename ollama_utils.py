import ollama
import tiktoken


# see: https://github.com/ollama/ollama/blob/main/docs/api.md
all_ollama_options = """{
    "num_keep": 5,
    "seed": 42,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "min_p": 0.0,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": true,
    "stop": ["\n", "user:"],
    "numa": false,
    "num_ctx": 1024,
    "num_batch": 2,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": false,
    "vocab_only": false,
    "use_mmap": true,
    "use_mlock": false,
    "num_thread": 8
  }"""



def query_llm(
        prompt,
        model='gpt-oss', #'gemma3:27b',
        num_predict=1000,
        num_tries=1,
        device='cuda', #'cpu'
        system_prompt=None, #'You are a helpful assistant'
        image_path=None,
):
    encoding = tiktoken.get_encoding('o200k_base')
    # encoding = tiktoken.get_encoding('cl100k_base')
    num_ctx = len(encoding.encode(prompt)) + num_predict #;print(num_ctx)
    options = {
        'temperature': 0,
        'num_predict': num_predict,  # 1000, #num_predict,
        'num_ctx': num_ctx,
    }
    if device == 'cpu': options['num_gpu'] = 0

    for _ in range(num_tries):
        try:
            response = ollama.generate(
                model = model,
                prompt = prompt,
                system = system_prompt,
                images = [image_path] if image_path else [],
                options = options,
            ).response
            break
        except Exception as e:
            print(e)

    return response


# def query_llm(
#         prompt,
#         model='gemma3:27b',
#         num_predict=1000,
#         num_tries=1,
#         device='cuda', #'cpu'
#         system_prompt=None, #'You are a helpful assistant'
# ):
#     encoding = tiktoken.get_encoding('cl100k_base')
#     num_ctx = len(encoding.encode(prompt)) + num_predict #;print(num_ctx)
#     options = {
#         'temperature': 0,
#         'num_predict': num_predict,  # 1000, #num_predict,
#         'num_ctx': num_ctx,
#     }
#     if device == 'cpu': options['num_gpu'] = 0

#     messages = []
#     if system_prompt is not None: messages += [
#         {'role': 'system', 'content': system_prompt},
#     ]
#     messages += [
#         {
#           'role': 'user',
#           'content': prompt,
#           # 'images': ['data/S-065_form/output-1.png', 'data/S-065_form/output-2.png'],
#         },
#     ]

#     for _ in range(num_tries):
#         try:
#             response = ollama.chat(
#                 model = model, #'gemma3:27b', #'llama3.3', #'llama3.1', 'llava',
#                 # format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
#                 messages=messages,
#                 options=options,
#                 # stream=True,
#             ).message.content
#             break
#         except Exception as e:
#             print(e)

#     return response


# def query_llm(
#         prompt,
#         model='gemma3:27b',
#         num_predict=256,
#         num_ctx=512,
# ):
#     response = ollama.chat(
#         model = model, #'gemma3:27b', #'llama3.3', #'llama3.1', 'llava',
#         # format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
#         messages=[
#         {
#           'role': 'user',
#           'content': prompt,
#           # 'images': ['data/S-065_form/output-1.png', 'data/S-065_form/output-2.png'],
#         },
#         ],
#         options={
#             'temperature': 0,
#             'num_predict': num_predict, #1000, #num_predict,
#             'num_ctx': num_ctx,
#         },
#         # stream=True,
#     ).message.content

#     return response


def parse_tagged_response(response, tags=['think', 'answer']): #, max_tokens=1000):
    parsed_info = {}

    for tag in tags:
        info = '' #None
        begin, end = f'<{tag}>', f'</{tag}>'

        if begin in response:
            info = response.split(begin)[1].strip()
            info = info.split(end)[0]

        parsed_info[tag] = info #clip_text(info, max_tokens) if info is not None else None

    return parsed_info


def parse_tag_from_response(response, tag='thought'):
    return parse_tagged_response(response, [tag])[tag]


def clip_text(text, max_tokens=1000, from_bottom=False):
    words = text.split()
    if from_bottom:
        clipped_text = ' '.join(words[-max_tokens:])
    else:
        clipped_text = ' '.join(words[:max_tokens])

    return clipped_text

