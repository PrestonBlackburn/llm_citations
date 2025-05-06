import random

EXAMPLE_DOCUMENT_1 = "DesignToolsGrading.md"
EXAMPLE_CHUNK_1 = """### Grading Criteria

The project is worth a maximum of 2 points. You can receive partial credit based on the completion of skills and graded criteria. Successful projects will meet the following graded criteria and expectation:

* Followed the standards for a memo with correspondence information. Responses are complete and detailed statements about the project outcomes. Writing is organized in a usable, useful, and enjoyable way. I understand how you planned, completed, and reflected on your work and lessons learned.	  
* Completed: The project closely mirrors the original document to replicate with appropriate updates and elements to demonstrate functional abilities with the tools under practice.	"""

EXAMPLE_DOCUMENT_2 = "DesignToolsChallenge.md"
EXAMPLE_CHUNK_2 = """### Requirements

Your work will be graded based on the following requirements:

1. A formal memo as your project retrospective that details:  
   1. *Work Assessment:* how you met the objectives of the project  
   2. *Process Reflection:* why you made your creative choices and solved problems during the project  
   3. *Applications:* how the skills, technology, and experiences from this project will help you improve as a professional in your field  
2. Replicated document that includes a close match for the header, headings, sidebars/columns, text box, diagram, fonts, colors, and icons.  
3. Your poster should look like the reference except for the colors all changed to be UCCS themed (black \#000000, gold \#cfb87c, white \#ffffff, and grey \#565a5c and  \#a2a4a3).

You must include polished writing, visuals, and design throughout the sections.
"""

SHORT_DOCUMENT_1 = "DesignToolsGrading.md"
SHORT_CHUNK_1 = """### Grading Criteria

The project is worth a maximum of 2 points. You can receive partial credit based on the completion of skills and graded criteria. Successful projects will meet the following graded criteria and expectation:"""
SHORT_DOCUMENT_2 = "DesignToolsChallenge.md"
SHORT_CHUNK_2 = """### Requirements

Your work will be graded based on the following requirements:

1. A formal memo as your project retrospective that details:  
   1. *Work Assessment:* how you met the objectives of the project  
   2. *Process Reflection:* why you made your creative choices and solved problems during the project  
You must include polished writing, visuals, and design throughout the sections.
"""


def get_inference_chunks(break_cache: bool = False) -> str:
    # Simulate chunks for inference (From RAG)
    rng = random.random()
    content_1 = f"""<content>
<content_source>
{ EXAMPLE_DOCUMENT_1 }
</content_source>

<content_chunk>
{ EXAMPLE_CHUNK_1 }
</content_chunk>

</content>
"""
    content_2 = f"""<content>
<content_source>
{ EXAMPLE_DOCUMENT_2 }
</content_source>

<content_chunk>
{ EXAMPLE_CHUNK_2 }
</content_chunk>

</content>
"""
    if break_cache:
        # if first tokens are cached, use rand to ignore the cache
        return str(rng) + "\n" + content_1 + content_2
    else:
        return content_1 + content_2


def get_inference_chunks_short() -> str:
    # shortening content chunks to try and speed up local inference + fit context window
    content_1 = f"""<content>
<content_source>
{ SHORT_DOCUMENT_1 }
</content_source>

<content_chunk>
{ SHORT_CHUNK_1 }
</content_chunk>

</content>
"""
    content_2 = f"""<content>
<content_source>
{ SHORT_DOCUMENT_2 }
</content_source>

<content_chunk>
{ SHORT_CHUNK_2 }
</content_chunk>

</content>
"""
    return content_1 + content_2
