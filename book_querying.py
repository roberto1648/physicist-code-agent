import os

import ollama_utils
import main_points_rag


CHAPTERS_DICT = {
    0: {'title': 'Introduction and Review',
    'filepath': 'data/book_chapters/ch00/ch00.rbtex'},
    1: {'title': 'Conservation of Mass',
    'filepath': 'data/book_chapters/ch01/ch01.rbtex'},
    2: {'title': 'Conservation of Energy',
    'filepath': 'data/book_chapters/ch02/ch02.rbtex'},
    3: {'title': 'Conservation of Momentum',
    'filepath': 'data/book_chapters/ch03/ch03.rbtex'},
    4: {'title': 'Conservation of Angular Momentum',
    'filepath': 'data/book_chapters/ch04/ch04.rbtex'},
    5: {'title': 'Thermodynamics',
    'filepath': 'data/book_chapters/ch05/ch05.rbtex'},
    6: {'title': 'Waves', 'filepath': 'data/book_chapters/ch06/ch06.rbtex'},
    7: {'title': 'Relativity', 'filepath': 'data/book_chapters/ch07/ch07.rbtex'},
    8: {'title': 'Atoms and Electromagnetism',
    'filepath': 'data/book_chapters/ch08/ch08.rbtex'},
    9: {'title': 'Circuits', 'filepath': 'data/book_chapters/ch09/ch09.rbtex'},
    10: {'title': 'Fields', 'filepath': 'data/book_chapters/ch10/ch10.rbtex'},
    11: {'title': 'Electromagnetism',
    'filepath': 'data/book_chapters/ch11/ch11.rbtex'},
    12: {'title': 'Optics', 'filepath': 'data/book_chapters/ch12/ch12.rbtex'},
    13: {'title': 'Quantum Physics',
    'filepath': 'data/book_chapters/ch13/ch13.rbtex'},
    14: {'title': 'Additional Topics in Quantum Physics',
    'filepath': 'data/book_chapters/ch14/ch14.rbtex'}
}


def query_book(
    query,
    chapters_dict=CHAPTERS_DICT,
    verbose=True,
):
    toc = get_toc(chapters_dict)
    chapter_number = get_chapter_number(query, toc, verbose)
    chapter_text = ''

    if chapter_number in chapters_dict:
        filepath = chapters_dict[chapter_number]['filepath']
        title = chapters_dict[chapter_number]['title']

        if os.path.exists(filepath):
            with open(filepath, 'r') as fp:
                chapter_text = fp.read()

    if verbose: print(f'Retrieved chapter {chapter_number}: {title}')

    if chapter_text.strip():
        answer, _ = main_points_rag.run(
            query=query,
            paper_text=chapter_text,
            model ='gpt-oss',
            num_predict =4000,
            num_tries = 3,
            threshold = 0.6,
            num_chunks_per_point =1,
            window_size =3,
            max_iterations =3,
            verbose = verbose,
        )

    if not answer.strip(): answer = 'No relevant information found.'

    return answer


def get_chapter_number(query, toc, verbose=False):
    prompt = (
        'Your task is to determine what is the most relevant book chapter for a given query '
        "The book's table of contents (toc) and the query are given below:\n\n"
        f'<toc>\n{toc}\n</toc>\n\n'
        f'<query>\n{query}\n</query>\n\n'
        'Provide your answer in the following format:\n\n'
        '<thoughts>\n'
        "Your reasoning on what is the query's topic and what book chapter may be the most related to it.\n"
        '</thoughts>\n\n'
        '<chapter_number>\n'
        'A single integer representing the chapter number. Do not write any other text here.'
        '\n</chapter_number>\n\n'
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
    number = ollama_utils.parse_tag_from_response(response, 'chapter_number')
    number = int(number.strip()) if number.strip().isdigit() else 0

    if verbose:
        print(prompt[:500], '\n\n...\n\n', prompt[-500:])
        print(response)

    return number


def get_toc(chapters_dict=CHAPTERS_DICT):
    return '\n'.join([f'Chapter {k}: {v["title"]}' for k, v in chapters_dict.items()])