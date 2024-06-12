from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from time import sleep
from pypdf import PdfReader
from openai import OpenAI
from anthropic import Anthropic
from pathlib import Path
import json
import re
import streamlit as st



pc = Pinecone(api_key = st.secrets["PINECONE_API_KEY"])
claude_client = Anthropic(api_key = st.secrets["CLAUDE_API_KEY"])
openai_client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

prompt_template = '''You are Anthropic's DocBot, a helpful assistant that is an expert at answering users\' questions with citations.

Here are the documents:
<documents>
{documents}
</documents>

When a user asks a question, perform the following tasks:
1. Find the quotes from the documentation that are the most relevant to answering the question. These quotes can be quite long if necessary. You may need to use many quotes to answer a single question.
2. Assign numbers to these quotes in the order they were found. Each segment of the documentation should only be assigned a number once.
3. Based on the document and quotes, answer the question. Directly quote the documentation when possible.
4. Avoid double quotes inside the content of relevant passages.
5. When answering the question provide citations references in square brackets containing the number generated in step 2 (the number the citation was found)
6. Structure the output in the following markdown format:
```
## Answer
A plain text answer, formatted as Markdown[1]

## Citations

### Citation 1
- name: string
- page: string
- relevant_passages: "string" // relevant passage in a document
...
```'''

def get_index(index_name='demand-foresight'):
    try:
        # connect to index
        print('loading document index...')
        index = pc.Index(index_name)
        sleep(1)
        return index
    except:
        print(f'Cannot find index {index_name}!')
        exit()

def get_embeddings(texts, model="text-embedding-3-small"):
    while True:
        try:
            embeddings = openai_client.embeddings.create(input=texts, model=model)
            embeddings = [d.embedding for d in embeddings.data]
            return embeddings
        except:
            sleep(5)

# Formatting search results
def format_results(extracted: list[str]) -> str:
    result = []
    for d in extracted['matches']:
        name = d['metadata']['name']
        page = d['metadata']['page']
        content = d['metadata']['content']
        result.append(f'<item name="{name}" page="{page}">\n<page_content>\n{content}\n</page_content>\n</item>')

    return '\n'.join(result)

def find_json_object(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')

    if start_index != -1 and end_index != -1:
        json_string = input_string[start_index:end_index+1]
        try:
            json_object = json.loads(json_string)
            return json_object
        except json.JSONDecodeError:
            print('cannot find json object from LLM response!')

    return None

def get_answer(completion):
    # Regex to extract answer from <answer> xml tags
    answer = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if answer is None:
        return ""
    return answer.group(1)

def get_citations(completion):
    # Regex to extract citations from <citations> xml tags
    citations = re.search(r'<citations>(.*?)</citations>', completion, re.DOTALL)
    if citations is None:
        return {}
    
    obj = find_json_object(citations.group(1))
    if obj is None:
        return {}
    return obj

def rag(question, model_id='gpt-4o', temperature=0.0):
    index = get_index()
    question_embed = get_embeddings([question])[0]
    
    print('retrieving relevant documents...')
    results = index.query(
        vector=question_embed, top_k=10, include_metadata=True
    )

    system_prompt = prompt_template.format(documents=format_results(results))
    # print('system_prompt:', system_prompt)

    starter_stub = '## Answer'
    full_response = starter_stub
    if 'claude' in model_id:
        print('sending request to claude...')
        with claude_client.messages.stream(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            system=system_prompt,
            temperature=temperature,
            messages=[
                {
                    "role": "user", 
                    "content": question
                }
            ]
        ) as stream:
            print(starter_stub)
            for text in stream.text_stream:
                print(text, end="", flush=True)
                st.write_stream(stream.text_stream)
                full_response += text
    elif 'gpt' in model_id:
        stream = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "assistant",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                st.write_stream(stream)
                full_response += chunk.choices[0].delta.content

    return full_response



st.set_page_config(page_title="Quickstart App")
st.title('demo')

#claude_api_key = st.sidebar.text_input('Claude API Key')
select_model = st.sidebar.selectbox(label="Select Model", options=["GPT", "Claude"], index=0, key="model_selection")
temp = st.sidebar.slider("Temperature", min_value=0.00 , max_value=1.0, step=0.01, key="temperature")

with st.form('my_form'):
  question = st.text_area('Enter text:', 'How has the development of AI impacted job markets and workforce dynamics in 2023?')
  submitted = st.form_submit_button('Submit')
  if select_model == "GPT" and submitted:  
    st.write("You selected GPT. Execute GPT related operations here.")
    rag(question, model_id = "gpt", temperature = temp)
  # 在這裡執行與 GPT 相關的操作
  elif select_model == "Claude" and submitted:
    st.write("You selected Claude. Execute Claude related operations here.")
    rag(question, model_id = "claude", temperature = temp)