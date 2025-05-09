from __future__ import annotations
from pydantic import create_model
from enum import Enum
from typing import List, Union, Dict
from jinja2 import Template
import asyncio
from chunks import (
    EXAMPLE_DOCUMENT_1,
    EXAMPLE_DOCUMENT_2,
    EXAMPLE_CHUNK_1,
    EXAMPLE_CHUNK_2,
)
from prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHORT
)

import logging

_logger = logging.getLogger(__name__)

async def create_document_enum_cls(documents: List[str]) -> "DocumentEnum":
    # dynamically generate enum classes based on documents returned from rag
    enum_cls = Enum(
        "DocumentEnum", 
        {f"document_{str(doc_idx+1)}": doc for doc_idx, doc in enumerate(documents)}, 
        type = str
    )
    return enum_cls

async def create_citation_response_cls(document_enum: "DocumentEnum") -> "ResponseWithCitations":
    citation_cls = create_model(
        "ResponseWithCitations",
        response = (str,...),
        citations = (List[Union[document_enum, None]],None)
    )
    # ex - ResponseWithCitations(response="answer[1]", citations=[DocumentEnum.doc_1])
    return citation_cls

async def create_context_examples() -> List[Dict[str,str]]:
    # creating examples using same format as query
    DocumentEnumExample = await create_document_enum_cls(["Project_description.txt", "Assignment_sheet.txt"])
    ResponseWithCitationsExample = await create_citation_response_cls(DocumentEnumExample)
    _logger.debug(f"Document Enum Dynamic Class - {vars(DocumentEnumExample)}")
    _logger.debug(f"Citation Example Dynamic Class - {ResponseWithCitationsExample}")

    example_1 = ResponseWithCitationsExample(
        response="You must write a memo detailing your work assessment, reflections, and professional applications[1]. You also need to replicate the design document[2].",
        citations=[DocumentEnumExample.document_1, DocumentEnumExample.document_2],
    )
    example_2 = ResponseWithCitationsExample(
        response="Projects are graded based on writing quality, completeness, organization, and how well the work mirrors the original document[2]. Partial credit is available based on these criteria[1].",
        citations=[
            DocumentEnumExample.document_2,
            DocumentEnumExample.document_1,
            DocumentEnumExample.document_2,
        ],
    )
    # seems to already struggle to provide citations, so probably don't need negative examples.
    example_3 = ResponseWithCitationsExample(
        response="my answer that does not reference any of the provided documents. There are no parenthetical references.",
        citations=[None],
    )

    examples = [
        example_1,
        example_2, 
    #    example_3
    ]

    few_shot_dicts = []
    query = [
        "What are the project requirements?",
        "How is the project graded?"
    ]
    for idx, example in enumerate(examples):
        few_shot_dicts.append({
            "role": "user",
            "content": query[idx]
        })
        few_shot_dicts.append({
            "role": "assistant",
            "content": example.model_dump_json()
        })

    return few_shot_dicts



async def format_rag_docs(documents: List[Dict[str, str]]) -> str:
    # documents looks like: [{"content_source": "", "content_chunk": "", Other metadata...}]
    document_context_template = """{% for document_info in document_info_list %}<content>
        {% for key, value in document_info.items() %}
        <{{key}}>
            {{ value }}
        </{{key}}>
        {% endfor %}
    </content>
    {% endfor %}

    """
    document_context_header = """Below are the class documents related to the user's question that should be cited inline (ex: [1]):\n"""
    template = Template(document_context_template)
    document_context_str = template.render(document_info_list = documents)

    return document_context_header + document_context_str


async def get_structures_for_rag_docs(documents: List[Dict[str,str]]) -> "ResponseWithCitations":

    document_names = []
    for document in documents:
        for key, value in document.items():
            if key == "content_source":
                document_names.append(value)
    assert document_names != [], f"No documents provided with 'content_source' key! - check your inputs to fomrat_rag_docs:\n {documents}"

    DocumentEnum = await create_document_enum_cls(document_names)
    ResponseWithCitations = await create_citation_response_cls(DocumentEnum)

    return ResponseWithCitations


async def query_vector_db(user_query:str, user_uuid:str) -> List[Dict[str, str]]:

    placeholder_rag_data = [
        {"content_source": EXAMPLE_DOCUMENT_1, "content_chunk": EXAMPLE_CHUNK_1}, 
        {"content_source": EXAMPLE_DOCUMENT_2, "content_chunk": EXAMPLE_CHUNK_2}
    ]

    return placeholder_rag_data

async def mock_query_vector_db(user_query:str, user_uuid:str) -> List[Dict[str, str]]:

    rag_data = [
        {"content_source": EXAMPLE_DOCUMENT_1, "content_chunk": EXAMPLE_CHUNK_1}, 
        {"content_source": EXAMPLE_DOCUMENT_2, "content_chunk": EXAMPLE_CHUNK_2}
    ]

    return rag_data


# what happens if we get more context than just the documents + content?
async def get_context(documents: List[Dict[str, str]]) -> str:
    # documents looks like: [{"content_source": "", "content_chunk": "", Other metadata...}]
    
    # chunks (dynamic)
    document_context_str = await format_rag_docs(documents)

    context = SYSTEM_PROMPT + document_context_str

    return context

async def get_few_shot_examples() -> List[Dict[str, str]]:
    # examples (static)
    examples_context = await create_context_examples()

    return examples_context
    
