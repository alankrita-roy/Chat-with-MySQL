import streamlit as st
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from langchain_groq import ChatGroq
import pandas as pd
import re
import sqlparse

st.set_page_config(page_title="Chat with MySQL", page_icon="🦜")
st.title("🦜Chat with MySQL Database")

mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
mysql_port = st.sidebar.text_input("Port", value="3306")
mysql_db = st.sidebar.text_input("Database Name")
mysql_user = st.sidebar.text_input("MySQL Username", value="root")
mysql_password = st.sidebar.text_input("MySQL Password", type="password")
#api_key = st.sidebar.text_input("Groq API Key", type="password")
api_key = st.secrets["GROQ_API_KEY"]

if not (mysql_host and mysql_port and mysql_db and mysql_user and mysql_password):
    st.info("Please enter all MySQL connection details.")

if mysql_host and mysql_port and mysql_db and mysql_user and mysql_password:

    encoded_password = quote_plus(mysql_password)

    @st.cache_resource(ttl="2h", show_spinner=False)
    def configure_mysql_db(user, password, host, port, db):
        engine = create_engine(f"mysql+mysqlconnector://{user}:{quote_plus(password)}@{host}:{port}/{db}")
        return SQLDatabase(engine), engine

    db, engine = configure_mysql_db(mysql_user, mysql_password, mysql_host, mysql_port, mysql_db)

    full_schema = ""

    with engine.connect() as conn:
        tables = db.get_usable_table_names()
        st.sidebar.markdown("### 📋 Tables in Database")
        for table in tables:
            st.sidebar.write(f"- {table}")

        st.sidebar.markdown("### 🧩 Table Columns")
        for table in tables:
            columns = conn.execute(text(f"DESCRIBE {table} ")).fetchall()
            cols = [col[0] for col in columns]
            st.sidebar.write(f"**{table}**: {', '.join(cols)}")
            full_schema += f"Table: {table}\nColumns: {', '.join(cols)}\n"

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="gemma2-9b-it",
        streaming=True,
        max_tokens=1024
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=1
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your MySQL database?"}]
        st.session_state["last_df"] = None
        st.session_state["last_sql"] = None
        st.session_state["last_error"] = ""
        st.session_state["download_clicked"] = False

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.sidebar.button("💡 Suggest Sample Prompt"):
        st.session_state["sample_prompt"] = "Show me the top 5 products by total sales."

    user_query = st.chat_input("Ask your MySQL database...", key="sample_prompt")

    def ask(input_text: str) -> str:
        try:
            raw_response = agent.run(input_text)
            raw_response = re.sub(r"Could not parse LLM output: `+", "", raw_response).strip()
            return raw_response
        except Exception as e:
            response = str(e)
            if response.startswith("Could not parse LLM output"):
                return "❗ The assistant could not understand the response. Please try rephrasing your question."
            return response

    if user_query:

        safe_user_query = re.sub(r'[^\w\s,.!?=><()*%-]', '', user_query)

        SYSTEM_PROMPT = (
            f"You are an expert MySQL assistant. The following is the database schema you must strictly adhere to:\n"
            f"{full_schema}\n"
            "Your task is to generate syntactically correct MySQL queries based strictly on the exact columns and tables provided."
            " Respond with ONLY the pure SQL query. Do not include explanations, labels, or prefixes like 'Action:'."
        )

        full_query = SYSTEM_PROMPT + "\n" + safe_user_query

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.empty())

            response = ask(full_query)
            cleaned_response = re.sub(r"[`\n]+", " ", response).strip()
            cleaned_response = re.sub(r"(?i)^Action:\s*db_sql_query\s*SQL:\s*", "", cleaned_response)

            sql_match = re.search(r"(?i)(SELECT|UPDATE|INSERT|DELETE)[\s\S]+?;", cleaned_response)
            proposed_sql = sql_match.group(0).strip() if sql_match else None

            st.session_state.messages.append({"role": "assistant", "content": response})

            if proposed_sql:
                formatted_sql = sqlparse.format(proposed_sql, reindent=True, keyword_case='upper')

                try:
                    with engine.begin() as conn:
                        if proposed_sql.strip().lower().startswith("select"):
                            result = conn.execute(text(proposed_sql))
                            df = pd.DataFrame(result.fetchall(), columns=result.keys())

                            st.session_state["last_df"] = df
                            st.session_state["last_sql"] = formatted_sql
                            st.session_state["last_error"] = ""

                            if not df.empty:
                                st.success("✅ Query ran successfully.")
                            else:
                                st.info("✅ Query ran but returned no results.")

                        else:
                            conn.execute(text(proposed_sql))
                            st.session_state["last_df"] = None
                            st.session_state["last_sql"] = formatted_sql
                            st.session_state["last_error"] = ""

                            st.success("✅ Non-select query executed successfully.")

                except Exception:
                    st.session_state["last_df"] = None
                    st.session_state["last_sql"] = formatted_sql
                    st.session_state["last_error"] = "❗ Query execution failed."

            else:
                if not st.session_state.get("download_clicked", False):
                    st.session_state["last_error"] = "The assistant could not generate a valid SQL query. Please try rephrasing your question."

    if st.session_state.get("last_sql"):
        st.subheader("🔍 SQL Query")
        st.code(st.session_state["last_sql"], language="sql")

    if st.session_state.get("last_df") is not None:
        st.subheader("📊 Query Results (Top 5 Rows)")
        st.dataframe(st.session_state["last_df"].head(5), use_container_width=True)

        csv = st.session_state["last_df"].to_csv(index=False).encode('utf-8')

        if st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="query_result.csv",
            mime="text/csv",
            key="download"
        ):
            st.session_state["download_clicked"] = True

    if st.session_state.get("last_error") and not st.session_state.get("download_clicked", False):
        st.warning(st.session_state["last_error"])

    if st.session_state.get("download_clicked", False):
        st.session_state["last_error"] = ""
        st.session_state["download_clicked"] = False
