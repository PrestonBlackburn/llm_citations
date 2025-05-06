# Testing unstructured vs structured responses

import os
import asyncio
import time
from typing import List, Union
import re

from enum import Enum
from pydantic import BaseModel
from openai import AsyncOpenAI

# outlines + transformers for CFG
import outlines
from outlines import models
from outlines.models.openai import OpenAIConfig
from outlines.models.transformers import Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# documents from "RAG"
from chunks import (
    get_inference_chunks,
    get_inference_chunks_short,
)


# source documents from RAG
class DocumentEnumExample(str, Enum):
    document_1 = "Project_description.txt"
    document_2 = "Assignment_sheet.txt"
    none = None


# source documents from RAG
class DocumentEnum(str, Enum):
    document_1 = "DesignToolsGrading.md"
    document_2 = "DesignToolsChallenge.md"
    none = None


# one option to return citations
class ResponseWithCitationsExample(BaseModel):
    response: str
    citations: List[Union[DocumentEnumExample, None]]


class ResponseWithCitations(BaseModel):
    response: str
    citations: List[Union[DocumentEnum, None]]


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def parse_sources(text) -> List[List[int]]:
    # Example parsing the responses for the cited docs
    pattern = r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]"
    matches = re.findall(pattern, text)
    # ex: ['[1]', '[2, 3]', '[]', '[10,20,30]']
    match_ints = []
    for match in matches:
        match_list = match.split(",")
        match_vals = []
        for match_val in match_list:
            print(match_val)
            try:
                match_vals.append(int(match_val.replace("[", "").replace("]", "")))
            except:
                pass
        if match_vals != []:
            match_ints.append(match_vals)
    return match_ints


def get_prompt_template(break_cache: bool = False) -> str:
    content = get_inference_chunks(break_cache=break_cache)
    example_1 = ResponseWithCitationsExample(
        response="my answer that references the project description text file[1]. Continued response that references sssignment sheet[2].",
        citations=[DocumentEnumExample.document_1, DocumentEnumExample.document_2],
    )
    example_2 = ResponseWithCitationsExample(
        response="my answer that references the project description text file[1]. Continued response that references the sssignment sheet[2]. Yet another answer that references Example_Project_description.txt[1].",
        citations=[
            DocumentEnumExample.document_1,
            DocumentEnumExample.document_2,
            DocumentEnumExample.document_1,
        ],
    )
    example_2 = ResponseWithCitationsExample(
        response="my answer that references the project description text file[1]. Continued response that references the sssignment sheet[2]. Yet another answer that references Example_Project_description.txt[1].",
        citations=[
            DocumentEnumExample.document_1,
            DocumentEnumExample.document_2,
            DocumentEnumExample.document_1,
        ],
    )

    example_3 = ResponseWithCitationsExample(
        response="my answer that does not reference any of the provided documents. There are no parenthetical references.",
        citations=[None],
    )

    template = f"""<context> 
{content}
</context>

Always cite your sources using parenthetical referencing (ex - [1]) to references the document based on the order that it is returned in the citations. The references should always be an integer correspoding to the index of the citations list.

Full Example 1:
{ example_1.model_dump_json() }

Full Example 2:
{ example_2.model_dump_json() }

Full Example 3:
{ example_3.model_dump_json() }

"""

    return template


async def simple_query(user_query: str, break_cache: bool = False) -> tuple:
    prompt_context = get_prompt_template(break_cache=break_cache)
    stream = await client.responses.create(
        model="gpt-4.1-mini", instructions=prompt_context, input=user_query, stream=True
    )
    text_stream = ""
    start_time = time.perf_counter()
    first_token_time = None
    completion_tokens = 0
    async for event in stream:
        if event.type == "response.output_text.delta":
            if first_token_time is None:
                first_token_time = time.perf_counter()
            text_stream += event.delta
            completion_tokens += 1
    end_time = time.perf_counter()

    total_time = end_time - start_time
    ttft = first_token_time - start_time

    print("Final Response: ", text_stream)

    return total_time, ttft, completion_tokens


async def structured_query(user_query: str, break_cache: bool = False) -> tuple:

    prompt_context = get_prompt_template(break_cache=break_cache)
    async with client.beta.chat.completions.stream(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt_context},
            {"role": "user", "content": user_query},
        ],
        response_format=ResponseWithCitations,
    ) as stream:
        text_stream = ""
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = 0
        async for event in stream:
            # print(event)
            if event.type == "content.delta":
                if event.parsed is not None:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    # we can stream the json reponse, but will be more difficult to parse incrementally
                    text_stream += event.delta
                    completion_tokens += 1
            elif event.type == "content.done":
                pass
                # print("content.done")
            elif event.type == "error":
                print("Error in stream:", event.error)

    # we need final completion to actually parse results
    final_completion = await stream.get_final_completion()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = first_token_time - start_time
    # print("Final Parsed Citations: ", final_completion.choices[0].message.parsed.citations)
    # print("Final Parsed Citation Type: ", type(final_completion.choices[0].message.parsed.citations[0]))
    print("Text Stream results: ", text_stream)

    return total_time, ttft, completion_tokens


def get_context_and_grammar_template():
    content = get_inference_chunks_short()
    template = f"""<context> 
    {content}
    </context>

    Always cite your sources using parenthetical referencing followed by the referenced document name (ex - [1]<reference_document.txt>). The document name is provided in the <content_source> section along with the relvant content of the document.

    Example: The project must include a working demo [1]<reference_document.txt>. It must follow the assignment format [2]<assignment_info.txt>.
    """
    return template


def get_grammar(documents: List[str]) -> str:

    documents = [f'"{document}"' for document in documents]
    document_options = " | ".join(documents)

    citation_grammar = f"""
start: sentence+

sentence: TEXT citation TEXT*

citation: "[" NUMBER "]" "<" DOCUMENT ">"

%import common.NUMBER
%import common.WS
%import common.LETTER
%import common.WORD
%ignore WS

DOCUMENT: {document_options}
TEXT: /[^\\[]+/
"""
    return citation_grammar


def outlines_query(user_query: str) -> None:
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")
    # Load model and tokenizer (assuming ran get_models.py)
    local_dir = "/models/phi-3"
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # Load model with proper device and dtype
    torch_dtype = torch.float16
    if device.type == "cuda":
        device_map = "auto"
    else:
        device_map = "auto"  # Defaults to CPU

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, device_map=device_map, torch_dtype=torch_dtype
    )

    wrapped_model = Transformers(model, tokenizer)

    documents = ["Project_description.txt", "Assignment_sheet.txt"]
    citation_grammar = get_grammar(documents)
    generator = outlines.generate.cfg(wrapped_model, citation_grammar)
    # generator = outlines.generate.grammar(model, tokenizer, citation_grammar)

    prompt_context = get_context_and_grammar_template()

    full_response = ""
    start = time.time()
    stream = generator.stream(
        f"{prompt_context} \nUser Query: {user_query}", max_tokens=256
    )

    for token in stream:
        print(token, flush=True)
        full_response += token

    end = time.time()
    print(f"FULL RESPONSE: {full_response}")
    print(f"TOTAL TIME (s): {end - start}")


if __name__ == "__main__":

    test_type = "simple_query"
    print(f"RUNNING WITH TEST: {test_type}")
    user_query = """What is the grading criteria for the project? Cite your sources, for example [1]<Project_description.txt>."""
    user_query_not_in_sources = """What is the temperature of the sun?"""

    n_tests = 2
    n_total_time = 0
    n_ttft = 0
    n_completion_tokens = 0
    for i in range(0, n_tests):
        if test_type == "simple_query":
            total_time, ttft, completion_tokens = asyncio.run(
                simple_query(user_query, break_cache=True)
            )
        if test_type == "structured_query":
            total_time, ttft, completion_tokens = asyncio.run(
                structured_query(user_query, break_cache=True)
            )
        n_total_time += total_time
        n_ttft = ttft
        n_completion_tokens += completion_tokens

    avg_total_time = n_total_time / n_tests
    avg_ttft = n_ttft / n_tests
    avg_completion_tokens = n_completion_tokens / n_tests
    print(f"RESULTS: {avg_total_time}, {avg_ttft}, {avg_completion_tokens}")

    # local model takes a while to run
    # outlines_query(user_query)
