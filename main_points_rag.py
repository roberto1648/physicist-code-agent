import re
import ollama_utils
import chromadb_utils
import chunking


def run(
    query: str,
    paper_text: str,
    model: str ='gpt-oss',
    num_predict: int =4000,
    num_tries: int =1,
    threshold: float =0.4,
    num_chunks_per_point: int =1,
    window_size: int =3,
    max_iterations: int =3,
    verbose: bool =False,
) -> tuple[str, list[str]]:
    """
    Answer a query using only the information within a given paper text.
    A large language model (LLM) is given the query and paper text to find
    points of information explicitly appearing in the text that together can
    answer the full query or at least aspects of it. The embedded representation
    of the extracted points are then compared to paper chunk embeddings.
    Only chunks whose distance to a point is less than a threshold are selected.
    The text of each selected chunk is augmented with surrounding chunks
    determined by a window size. The selected and augmented chunks are then
    used to build context information. The context information is then given
    to an LLM to answer the query. The LLM answer is verified for accuracy
    with respect to the context information. If there are any unsupported claims
    the answer is refined (fixed). This process continues until there are
    no unsupported claims or until a maximum number of iterations is reached.
    Then the final answer and retrieved chunks are returned.
    :param query: The query to be answered.
    :param paper_text: The supporting information to answer the query.
    :param model: The name of an ollama model that will be used as LLM.
    :param num_predict: The maximum number of tokens in the LLM response.
    :param num_tries: The number of attempts to try to run the ollama model.
    :param threshold: The maximum semantic distance between an extracted point and a paper chunk.
    :param num_chunks_per_point: The maximum number of similar chunks to extract from each point.
    :param window_size: A window around each chunk by which each chunk is augmented to provide more context.
    :param max_iterations: The maximum number of answer/verification/fixing steps to perform.
    :param verbose: Whether or not to print the progress as each part of the algorithm completes.
    :return:
    answer: The answer to the query based on the provided paper text.
    retrieved_chunks: Text chunks from the paper that are selected and used to answer the query.
    """
    points = get_points_from_paper_and_query(paper_text, query, verbose)
    paper_chunks = chunking.get_chunks_from_document(
        text=paper_text,
        max_words=100,
        overlap=0, #0.2,
        min_paragraph_words=10,
        paragraph_sep='\n\n',
        sentense_sep='.',
        max_iterations=100,
    )
    retrieved_chunks = get_chunks_similar_to_points(
        points=points,
        chunks=paper_chunks,
        threshold=threshold, #0.4,
        num_chunks_per_point=num_chunks_per_point, #1,
        window_size=window_size, #3,
    )
    answer = answer_query(
        query=query,
        chunks=retrieved_chunks,
        model=model, #'gpt-oss',
        num_predict=num_predict, #4000,
        num_tries=num_tries, #1,
        verbose=verbose, #False,
    )

    for iteration_number in range(max_iterations):
        if answer.strip():
            unsupported_claims = verify_answer(
                query=query,
                chunks=retrieved_chunks,
                answer=answer,
                model=model, #'gpt-oss',
                num_predict=num_predict, #4000,
                num_tries=num_tries, #1,
                verbose=verbose, #False,
            )

            if not unsupported_claims:
                break
            else:
                answer = fix_answer(
                    query=query,
                    chunks=retrieved_chunks,
                    answer=answer,
                    unsupported_claims=unsupported_claims,
                    model=model, #'gpt-oss',
                    num_predict=num_predict, #4000,
                    num_tries=num_tries, #1,
                    verbose=verbose, #False,
                )

    return answer, retrieved_chunks


def get_points_from_paper_and_query(paper_text, query, verbose=False):
    prompt = (
        'Your task is to extract the most relevant points of information from a paper '
        'in order to answer a query. '
        'The points must closely correspond to literal excerpts from the paper. '
        'Both the paper and the query are given below:\n\n'
        f'<paper>\n{paper_text}\n</paper>\n\n'
        f'<query>\n{query}\n</query>\n\n'
        'Provide your answer in the following format:\n\n'
        '<thoughts>\n'
        'Your reasoning on what pieces of information are needed to answer the query. '
        'Are any of those pieces of information present in the paper? Which ones?\n'
        '</thoughts>\n\n'
        '<points>\n'
        'A list of one-sentence distinct points of information from the paper (if any) that are relevant to the query. '
        'In as much as possible, the points must be independent from each other. '
        'Example:\n- point 1.\n- point 2.\n...'
        '\n</points>\n\n'
    )  # ;print(prompt[:500], '\n...\n', prompt[-500:])

    # system_prompt = (
    #     'Your are a smart assistant capable of extracting the most important points of information from a paper. '
    #     'Provide your answers within XML fields as indicated, do not write text outside of the XML fields.'
    # )

    response = ollama_utils.query_llm(
        prompt=prompt,
        model='gpt-oss',
        num_predict=4000,  # 2000, #1000,
        num_tries=1,
        device='cuda',  # 'cpu'
        system_prompt=None,  # system_prompt, #None, #'You are a helpful assistant'
        image_path=None,
    )
    points = ollama_utils.parse_tagged_response(response, ['points'])['points']
    points = re.split('-\s+(.*)\n+', points)
    points = [p.strip() for p in points if p.strip()]

    if verbose:
        print(prompt[:500], '\n\n...\n\n', prompt[-500:])
        print(response)

    return points


def get_chunks_similar_to_points(
        points,
        chunks,
        threshold=0.4,
        num_chunks_per_point=1,
        window_size=3,
):
    retrieved_chunks = []
    distances = []
    chunk_idxs = []
    half_window = window_size // 2

    for point in points:
        answer_obj = chromadb_utils.query_chunks(
            query=point,
            chunks=chunks,
            n_results=num_chunks_per_point,  # 1, #4,
            # collection_name='my_collection',
            # chroma_client=chroma_client, # comment this outside
        )  # ;print(answer_obj)
        distances_to_point = answer_obj['distances'][0]  # ;print(distances)

        for k, d in enumerate(distances_to_point):
            if d < threshold:
                # print(k, d)
                chunk_idx = int(answer_obj['ids'][0][k].replace('chunk_', ''))

                if chunk_idx not in chunk_idxs:
                    txt = ' '.join(chunks[chunk_idx - half_window: chunk_idx + half_window + 1])
                    retrieved_chunks.append(txt)
                    distances.append(d)
                    chunk_idxs.append(chunk_idx)

    retrieved_chunks = [chunk for chunk in retrieved_chunks if chunk.strip()]

    return retrieved_chunks


def answer_query(
        query,
        chunks,
        model='gpt-oss',
        num_predict=4000,
        num_tries=1,
        verbose=False,
):
    information = '\n\n'.join(chunks)
    prompt = (
        'Your task is to answer a query based only on provided information. '
        'Both the query and information are given below.\n\n'
        f'<query>\n{query}\n</query>\n\n'
        f'<information>\n{information}\n</information>\n\n'
        'Provide your answer in the following format:\n\n'
        '<thoughts>\n'
        'Your reasoning on what information persent in the provided information can be used to answer the query, '
        'if at all possible.'
        '\n</thoughts>\n\n'
        '<is-answerable>\n'
        'Is there enough information to answer the query? write down "Yes" or "No"'
        '\n<is-answerable>\n\n'
        '<answer>\n'
        'Your answer to the query based only on the provided information. Leave empty if no enough information was provided.'
        '\n</answer>'
    )
    response = ollama_utils.query_llm(
        prompt=prompt,
        model=model,  # 'gpt-oss',
        num_predict=num_predict,  # 4000, #2000, #1000,
        num_tries=num_tries,  # 1,
        device='cuda',  # 'cpu'
        system_prompt=None,  # system_prompt, #None, #'You are a helpful assistant'
        image_path=None,
    )
    answer = ollama_utils.parse_tag_from_response(response, 'answer').strip()

    if verbose:
        print(prompt[:500], '\n\n...\n\n', prompt[-500:])
        print(response)

    return answer.strip()


def verify_answer(
        query,
        chunks,
        answer,
        model='gpt-oss',
        num_predict=4000,
        num_tries=1,
        verbose=False,
):
    information = '\n\n'.join(chunks)
    prompt = (
        "Your task is to verify whether an assistant's answer to a query is accurately based only "
        "the explicitly provided context information. The context information, query, and answer are given below.\n\n"
        f"<context-information>\n{information}\n</context-information>\n\n"
        f"<query>\n{query}\n</query>\n\n"
        f"<assistant-answer>\n{answer}\n</answer>\n\n"
        "Provide your answer in the following format:\n\n"
        "<thoughts>\n"
        "Your thoughts regarding whether the answer is accurately derived from the context information. "
        "Are there any important points in the response that are not supported by the context information?"
        "\n</thoughts>\n\n"
        '<is-supported>\n'
        "Is the assistant's answer to the query supported by the context information? write down Yes or No"
        '\n<is-supported>\n\n'
        "<unsupported-claims>\n"
        "Write here any unsupported claims made by the assistant. Leave empty if there are no unsupported claims."
        "\n</unsupported-claims>\n\n"
    )

    response = ollama_utils.query_llm(
        prompt=prompt,
        model=model,  # 'gpt-oss',
        num_predict=num_predict,  # 4000, #2000, #1000,
        num_tries=num_tries,  # 1,
        device='cuda',  # 'cpu'
        system_prompt=None,  # system_prompt, #None, #'You are a helpful assistant'
        image_path=None,
    )
    unsupported_claims = ollama_utils.parse_tag_from_response(response, 'unsupported-claims')

    if verbose:
        print(prompt[:500], '\n\n...\n\n', prompt[-500:])
        print(response)

    return unsupported_claims.strip()


def fix_answer(
        query,
        chunks,
        answer,
        unsupported_claims,
        model='gpt-oss',
        num_predict=4000,
        num_tries=1,
        verbose=False,
):
    information = '\n\n'.join(chunks)
    prompt = (
        'Your task is to fix an answer to a query written by an assistant. '
        'The answer must be based only on provided context information. '
        "A reviewer has identified claims in the assistant's answer that are unsupported by the context information. "
        "The query, context information, assistant's answer, and the identified unsupported claims are given below.\n\n"
        f'<query>\n{query}\n</query>\n\n'
        f'<information>\n{information}\n</information>\n\n'
        f'<assistant-answer>\n{answer}\n</assistant-answer>\n\n'
        f'<unsupported-claims>\n{unsupported_claims}\n<unsupported-claims>\n\n'
        'Provide your answer in the following format:\n\n'
        '<thoughts>\n'
        "Your thoughts on what needs to be corrected in the assistant's answer."
        '\n</thoughts>\n\n'
        '<fixed-answer>\n'
        'A fixed answer to the query fully supported by the context information.'
        '\n</fixed-answer>'
    )

    response = ollama_utils.query_llm(
        prompt=prompt,
        model=model,  # 'gpt-oss',
        num_predict=num_predict,  # 4000, #2000, #1000,
        num_tries=num_tries,  # 1,
        device='cuda',  # 'cpu'
        system_prompt=None,  # system_prompt, #None, #'You are a helpful assistant'
        image_path=None,
    )
    fixed_answer = ollama_utils.parse_tag_from_response(response, 'fixed-answer')

    if verbose:
        print(prompt[:500], '\n\n...\n\n', prompt[-500:])
        print(response)

    return fixed_answer.strip()

