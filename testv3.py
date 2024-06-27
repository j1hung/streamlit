import streamlit as st
from langchain_conversational_rag import rag

# 设置页面配置和标题
st.set_page_config(page_title="Quickstart")
st.title('demo')

# 侧边栏选择模型
select_model = st.sidebar.selectbox(label="Choose model", options=["GPT", "Claude"], index=0, key="model_selection")

# 初始化会话状态
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
if 'latest_response' not in st.session_state:
    st.session_state.latest_response = ''

# 清除聊天记录的功能
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.session_id = None
    st.session_state.latest_response = ''

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# 显示聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户聊天输入和对应生成
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if select_model == "GPT":
                    session_id, stream = rag(prompt, model_id="gpt-4o", session_id=st.session_state.session_id)
                elif select_model == "Claude":
                    session_id, stream = rag(prompt, model_id="claude-3-5-sonnet-20240620", session_id=st.session_state.session_id)
                
                # 更新 session_id
                st.session_state.session_id = session_id
                
                # 处理并显示回答的流数据
                full_response = ''
                placeholder = st.empty()
                for item in stream:
                    if answer_chunk := item.get("answer"):
                        full_response += answer_chunk
                        placeholder.markdown(full_response)
                
                st.session_state.latest_response = full_response

        message = {"role": "assistant", "content": st.session_state.latest_response}
        st.session_state.messages.append(message)
        st.session_state.latest_response = ''
        st.experimental_rerun()