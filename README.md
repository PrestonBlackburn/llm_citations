# LLM Citations Exploration

A repo to do some digging into a few different options for citations. Citations can gives users more confidence in the model's response by making it easier to validate the response. We'll compare a few different options for citation generation, but all of the options will assume that a number of documents have been first retreived from a RAG workflow.

<br/>

The options for citation generation we'll test are:  
1. Vanilla plain text responses (and cross our fingers)  
2. Structured output for citations  
3. Custom context free grammar rules  

<br/>

See the associated blog post for more info: (TBD)   


## Testing the approaches

Get local model
```ps1
python get_models.py
```

run locally 
```ps1
python chat.py
```

As part of the process I found out that the outlines and guidance libraries can't enforce CFGs with the OpenAI API, since we need lower level access to the models inference results for token level grammar enforcement. Guidance can work around this with a soft constraint enforcement but isn't true CFG. I still wanted to test them out, so I ended up using a local model to test CFG with the outlines library.


## Approaches breakdown


### Simple Results
*didn't want to spend enough to do a more representative comparison*  

n = 10  

| Strategy          | Model        |  Avg TTFT (s) | Avg Response Time (s) | Avg Completion Tokens |  
| ------------------| ------------ | ------------- | --------------------- | ---------  |
| Standard          | gpt-4.1-mini | 0.039    | 2.89 |  124.0 |
| Structured Output | gpt-4.1-mini | 0.012    | 2.66 |  114.5 |

*Outlines doesn't support streaming for the OpenAI API yet*
<br/>