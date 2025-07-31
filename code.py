import streamlit as st
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from langchain_groq import ChatGroq
import pandas as pd
import re
import os
import sqlparse
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat with MySQL", page_icon="🦜")
st.title("🦜Chat with MySQL Database")

mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
mysql_port = st.sidebar.text_input("Port", value="3306")
mysql_db = st.sidebar.text_input("Database Name", value=os.getenv("DATABASE"))
mysql_user = st.sidebar.text_input("MySQL Username", value="root")
mysql_password = st.sidebar.text_input("MySQL Password", type="password", value=os.getenv("MYSQL_PASSWORD"))

api_key = os.getenv("GROQ_API_KEY")

# Connect button
if st.sidebar.button("🔌 Connect to Database"):
    if mysql_host and mysql_port and mysql_db and mysql_user and mysql_password:
        st.session_state["connect_clicked"] = True
    else:
        st.warning("⚠️ Please fill in all the MySQL connection details before connecting.")

if st.session_state.get("connect_clicked", False):
    encoded_password = quote_plus(mysql_password)

    @st.cache_resource(ttl="2h", show_spinner=False)
    def configure_mysql_db(user, password, host, port, db):
        engine = create_engine(f"mysql+mysqlconnector://{user}:{quote_plus(password)}@{host}:{port}/{db}")
        return engine

    engine = configure_mysql_db(mysql_user, mysql_password, mysql_host, mysql_port, mysql_db)

    full_schema = ""

    with engine.connect() as conn:
        result = conn.execute(text("SHOW TABLES")).fetchall()
        tables = [row[0] for row in result]

        st.sidebar.markdown("### 📋 Tables in Database")
        for table in tables:
            st.sidebar.write(f"- {table}")

        st.sidebar.markdown("### 🧩 Table Columns")
        for table in tables:
            columns = conn.execute(text(f"DESCRIBE {table}")).fetchall()
            cols = [col[0] for col in columns]
            st.sidebar.write(f"**{table}**: {', '.join(cols)}")
            full_schema += f"Table: {table}\nColumns: {', '.join(cols)}\n"

    # 💡 Suggest sample prompt
    if st.sidebar.button("💡 Suggest Sample Prompt"):
        st.session_state["suggested_prompt"] = "Show me the top 5 products by total sales."

    if "suggested_prompt" in st.session_state:
        st.info(f"💡 Suggested Prompt: *{st.session_state['suggested_prompt']}*")

    # Setup LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="gemma2-9b-it",
        streaming=True,
        max_tokens=1024
    )

    # Setup chat
    if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your MySQL database?"}]
        st.session_state["last_df"] = None
        st.session_state["last_sql"] = None
        st.session_state["last_error"] = ""
        st.session_state["download_clicked"] = False

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input("Ask your MySQL database...")

    def ask_llm(prompt: str):
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"❗ Error: {str(e)}"

    if user_query:
        safe_user_query = re.sub(r'[^\w\s,.!?=><()*%-]', '', user_query)

        SYSTEM_PROMPT = (
            f"You are an expert MySQL assistant. Use only the following schema:\n"
            f"{full_schema}\n\n"
            "Your job is to write a valid MySQL query based strictly on this schema. "
            "Respond with ONLY the SQL query, without any explanation, labels, or special prefixes (like Action:, SQL:, sql_db_list_tables, etc.). "
            "Do not include markdown, comments, or backticks. End the query with a semicolon."
        )

        full_prompt = SYSTEM_PROMPT + "\n" + safe_user_query

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            raw_sql_response = ask_llm(full_prompt)

            # Extract SQL
            sql_match = re.search(r"(?i)(SELECT|UPDATE|INSERT|DELETE)[\s\S]*?;", raw_sql_response)
            proposed_sql = sql_match.group(0).strip() if sql_match else None

            st.session_state.messages.append({"role": "assistant", "content": raw_sql_response})

            if proposed_sql:
                formatted_sql = sqlparse.format(proposed_sql, reindent=True, keyword_case='upper')

                try:
                    with engine.begin() as conn:
                        if proposed_sql.lower().startswith("select"):
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
                st.session_state["last_error"] = "❗ Could not extract a valid SQL query. Please rephrase your request."

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
