

def get_chunks_from_document(
    text,
    min_paragraph_words=10,
    paragraph_sep='\n\n',
    **kwargs,
):
    chunks = get_paragraphs(
        text,
        sep=paragraph_sep,
        min_words=min_paragraph_words,
    )
    chunks = deorphan_equations(chunks)

    return chunks


# def get_chunks_from_document(
#     text,
#     max_words=100,
#     overlap=0, #0.2,
#     min_paragraph_words=10,
#     paragraph_sep='\n\n',
#     sentense_sep='.',
#     max_iterations=100,
# ):
#     paragraphs = get_paragraphs(
#         text,
#         sep=paragraph_sep,
#         min_words=min_paragraph_words,
#     )
#     chunks = []
#
#     for paragraph in paragraphs:
#         new_chunks = get_chunks_from_paragraph(
#             paragraph,
#             max_words=max_words,
#             overlap=overlap,
#             sentence_sep=sentense_sep,
#             max_iterations=max_iterations,
#         )
#         chunks += new_chunks
#
#     return chunks


def get_chunks_from_paragraph(
    paragraph,
    max_words=100,
    min_words=5,
    overlap=0.2,
    sentence_sep='.',
    max_iterations=100,
):
    overlap = int(overlap * max_words)
    words = paragraph.split()
    chunks = []
    done = False
    pos = 0
    iteration = 0

    while not done:
        pi = pos - overlap
        pi = pi if pi >= 0 else 0 #;print(pi)
        pf = pi + max_words #;print(pf) ;print(words[pi: pf])

        chunk = ' '.join(words[pi:pf]) #;print(chunk_text)
        chunk_sentences = []
        current_sentence = ''

        for s in chunk.split(sentence_sep):
            current_sentence += s + sentence_sep

            if len(current_sentence.split()) < min_words:
                current_sentence += s
            else:
                chunk_sentences.append(current_sentence)
                current_sentence = ''
        
        if len(chunk_sentences) >= 3:
            si, sf = chunk_sentences[0], chunk_sentences[-1]
            pi += len(si.split())
            if pi < 0: pi = 0
            if not chunk.strip().endswith('.'): pf -= len(sf.split())
            chunk = ' '.join(words[pi:pf])

        chunks.append(chunk)
        pos = pf + 1 #;print(pos)
        iteration += 1 
        
        if (pos >= len(words)) or (iteration >= max_iterations): done = True

    return chunks


def get_paragraphs(text, sep='\n\n', min_words=10):
    splits = text.split(sep)
    paragraphs = []
    prepend = ''

    for p in splits:
        num_words = len([w.strip() for w in p if w.strip()])

        if num_words >= min_words:
            paragraphs.append(prepend + '\n\n' + p if prepend else p)
            prepend = ''
        else:
            prepend = p

    return paragraphs


def deorphan_equations(
        chunks=[
            'abcd ',
            r'ss ddss \begin{equation} ... \end{equation*}',
            r'ss ddss \begin{align} ... \end{align*}',
            r'ss ddss \begin{equation} kjsflksj'
            'chao'
            '...slkjlsjf \end{equation*}',
            r'ss ddss \begin{align} jaja'
            'hola',
            '...jiji \end{align*}',
        ],
):
    new_chunks = []
    current_chunk = ''

    for chunk in chunks:
        current_chunk += chunk
        cond = (r'\begin{eq' in chunk) and (r'\end{eq' not in chunk)
        cond |= (r'\begin{al' in chunk) and (r'\end{al' not in chunk)

        if not cond:
            new_chunks.append(current_chunk)
            current_chunk = ''

    return new_chunks