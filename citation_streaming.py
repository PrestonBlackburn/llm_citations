# Playing around with streaming responses, structure, and post processing for linking

import os
import asyncio
import time
from typing import List, Union, Callable, AsyncGenerator, Dict
import re
import logging
from openai import AsyncOpenAI
from openai.types.chat import ParsedChatCompletion

# few shot prompt construction
from few_shot_examples import (
    mock_query_vector_db, 
    get_context, 
    get_structures_for_rag_docs,
    get_few_shot_examples,
)

import logging

import few_shot_examples

_logger = logging.getLogger(__name__)


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def parse_plaintext_citations(text:str) -> List[str]:
    # Example parsing the responses for the cited docs
    pattern = r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]"
    matches = re.findall(pattern, text) # will block
    # ex: ['[1]', '[2, 3]', '[]', '[10,20,30]']
    return matches

async def get_citation_idx(citation: str) -> List[int]:
    match_list = citation.split(",")
    match_vals = []
    for match_val in match_list:
        print(match_val)
        try:
            match_vals.append(int(match_val.replace("[", "").replace("]", "")))
        except:
            pass
    # ex: '[2, 3]' -> [2, 3]
    return match_vals


async def parse_final_response(
        structured_response:"ResponseWithCitations", 
        document_map: dict
    ) -> str:
    # also need document names + chunks for reference and to be used in the template
    text = structured_response.response

    citations = await parse_plaintext_citations(text)

    for citation in citations:
        citation_idx = await get_citation_idx(citation)
        citation_idx_str = [str(citation_id) for citation_id in citation_idx]
        citation_text = f"""<sup class="text-blue-600 cursor-pointer citation" data-citation-id="citation-{"-".join(citation_idx_str)}">{citation}</sup>"""
        text = text.replace(citation, citation_text)

        file_name = "**DesignToolsGrading.md**"
        citation_content = """### Grading Criteria
The project is worth a maximum of 2 points. You can receive partial credit..."""
        reference = f"{file_name}\n{citation_content}"
        template = f"""<template id="citation-{"-".join(citation_idx_str)}"> {reference} </template>"""
        text += template

    return f"----FINAL PARSED RESPONSE----\n {text}"


# goal is to get something like:
# {"[1]": "doc1.txt", "[1, 2]": "doc1.txt, doc2.txt", "[3]": "doc3.txt"}
# then do a simple find and replace
# the only issue is if bracketed citations are already present, but probably ok...

# we'll return a html response to make it easy to construct the links
# <sup class="text-blue-600 cursor-pointer citation" data-citation-id="citation-1">[1]</sup> 
# and the citation info gets closed in a template tag:
#   <template id="citation-1">
#     **DesignToolsGrading.md**
#     ### Grading Criteria
    
#     The project is worth a maximum of 2 points. You can receive partial credit...
#   </template>


async def openai_structured_streamer(
        user_query: str, 
        prompt_context: str, 
        few_shot_examples: List[Dict[str, str]],
        ResponseFormat: "ResponseWithCitations"
        ) -> AsyncGenerator[Union[str, ParsedChatCompletion], None]:
    # Incrementally processes the streaming response to strip out json formatting
    expected_json_start = '{"response":'
    expected_json_end = 'citations=[<DocumentEnum' # we don't need to show the unformatted output
    buffer = expected_json_start
    stripped_response = False
    messages = []
    messages.append({"role": "system", "content": prompt_context})
    messages.extend(few_shot_examples)
    messages.append({"role": "user", "content": user_query})

    async with client.beta.chat.completions.stream(
        model="gpt-4.1-mini",
        messages=messages,
        response_format=ResponseFormat,
        temperature=0.0
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                if event.parsed is not None:
                    # we can stream the json reponse, but will be more difficult to parse incrementally
                    if not stripped_response:
                        if buffer.startswith(event.delta):
                            buffer = buffer[len(event.delta):]
                            continue
                        else:
                            stripped_response = True
                    if stripped_response:
                        yield event.delta

            elif event.type == "content.done":
                pass
                # print("content.done")
            elif event.type == "error":
                print("Error in stream:", event.error)

        
    # we need final completion to actually parse results
    final_completion = await stream.get_final_completion()
    # print("Final Parsed Citations: ", final_completion.choices[0].message.parsed.citations)
    # print("Final Parsed Citation Type: ", type(final_completion.choices[0].message.parsed.citations[0]))
    _logger.debug(f"FINAL COMPLETION: {final_completion}")
    yield final_completion

async def stream_aggregator(
        streamer: AsyncGenerator[Union[str, ParsedChatCompletion], None]
    ) -> AsyncGenerator[Union[str, "ResponseWithCitations"], None]:
    text_stream = ""
    async for token in streamer:
        if isinstance(token, str):
            text_stream += token
            yield text_stream
    
        if isinstance(token, ParsedChatCompletion):
            # last itteration is actually the structured output
            yield await parse_final_response(token.choices[0].message.parsed, {})


async def structured_query(
        user_query: str, 
        structured_streamer:Callable[
            [
                str, # user query
                str, # sys prompt
                List[Dict[str, str]], # few shot examples
                "ResponseWithCitations" # response format structure
            ], 
            AsyncGenerator[Union[str, ParsedChatCompletion], None]]
    ) -> AsyncGenerator[Union[str, "ResponseWithCitations"], None]:
    user_uuid = "a1"
    documents = await mock_query_vector_db(user_query, user_uuid)
    ResponseWithCitations = await get_structures_for_rag_docs(documents)
    context = await get_context(documents)
    few_shot_examples = await get_few_shot_examples()

    streamer = structured_streamer(user_query, context, few_shot_examples, ResponseWithCitations)
    aggregator = stream_aggregator(streamer)
    async for event in aggregator:
        print(event)
        # yield event

# Also, if no citations show up anywhere, append citations to end of text based on RAG docs
# Or maybe just note all of the referenced files somewhere else on the page?


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    user_query = "What is the grading criteria for the design tools project?"
    asyncio.run(structured_query(user_query, openai_structured_streamer))