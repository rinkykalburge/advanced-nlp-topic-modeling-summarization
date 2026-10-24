import streamlit as st
from nlp_core.documents import extract_text, split_documents
from nlp_core.preprocessing import preprocess_text
from nlp_core.topics import run_lda, topic_coherence, run_bertopic
from nlp_core.summarization import textrank_summary, transformer_summary
from nlp_core.keywords import extract_keywords
from nlp_core.evaluation import rouge_scores, compression_stats
from nlp_core.storage import init_db, authenticate, create_user, save_analysis, get_history, get_users, log_event
from nlp_core.reporting import build_report_pdf
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetimen

st.set_page_config(page_title="NLP Insight Studio", page_icon="🧠", layout="wide")
init_db()

if "user" not in st.session_state:
    st.session_state.user = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#0f766e); color:white;}
.small-note {font-size: .9rem; color: #64748b;}
div[data-testid="stMetric"] {border:1px solid #e2e8f0; padding:14px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🧠 NLP Insight Studio</h1><p>Topic discovery · Summarization · Document intelligence</p></div>', unsafe_allow_html=True)
st.caption("Analyze documents locally with classical NLP methods and optional transformer models.")

with st.sidebar:
    st.header("Account")
    if st.session_state.user:
        st.success(f"Signed in as {st.session_state.user['username']} ({st.session_state.user['role']})")
        if st.button("Sign out"):
            st.session_state.user = None
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Sign in", "Register"])
        with tab_login:
            lu = st.text_input("Username", key="login_user")
            lp = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign in", use_container_width=True):
                user = authenticate(lu, lp)
                if user:
                    st.session_state.user = user
                    log_event(lu, "login")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with tab_register:
            ru = st.text_input("Choose username", key="reg_user")
            rp = st.text_input("Choose password (8+ characters)", type="password", key="reg_pass")
            if st.button("Create account", use_container_width=True):
                ok, msg = create_user(ru, rp)
                if ok:
                    st.success(msg + " You can now sign in.")
                else:
                    st.error(msg)
        st.info("You can analyze documents without an account. Sign in to save analysis history.")
    st.divider()
    page = st.radio("Navigate", ["Home & Analyze", "Topic Modeling", "Summarization", "Evaluation", "Multi-Document", "History", "Admin", "About"])
    st.divider()
    st.caption("Privacy: files are processed by this app. Transformer downloads may contact Hugging Face when enabled.")

def load_inputs():
    uploads = st.file_uploader("Upload TXT, PDF, or DOCX files", type=["txt","pdf","docx"], accept_multiple_files=True)
    pasted = st.text_area("Or paste text here", height=180, placeholder="Paste an article, report, or research-paper text…")
    docs = []
    for up in uploads or []:
        try:
            txt = extract_text(up.name, up.getvalue())
            if txt.strip():
                docs.append({"name": up.name, "text": txt})
        except Exception as e:
            st.warning(f"Could not read {up.name}: {e}")
    if pasted.strip():
        docs.append({"name": "pasted_text.txt", "text": pasted.strip()})
    return docs

def render_analysis(docs):
    if not docs:
        st.warning("Upload at least one supported document or paste some text.")
        return
    if len(docs) > 1:
        st.info(f"{len(docs)} documents loaded. The combined corpus will be used for shared topic analysis.")
    combined = "\n\n".join(d["text"] for d in docs)
    cleaned, tokens = preprocess_text(combined)
    st.session_state.analysis = {"docs": docs, "combined": combined, "cleaned": cleaned, "tokens": tokens}
    c1,c2,c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Words", f"{len(combined.split()):,}")
    c3.metric("Cleaned tokens", f"{len(tokens):,}")
    with st.expander("Preview extracted text and preprocessing", expanded=False):
        st.text_area("Combined text preview", combined[:6000], height=180, disabled=True)
        st.text_area("Preprocessed preview", cleaned[:4000], height=140, disabled=True)
    if st.session_state.user and st.button("Save uploaded text to my analysis history"):
        save_analysis(st.session_state.user["id"], "document_upload", "Loaded " + str(len(docs)) + " document(s)", combined[:50000])
        st.success("Saved to history.")

if page == "Home & Analyze":
    st.subheader("1. Add your documents")
    docs = load_inputs()
    if st.button("Prepare documents", type="primary"):
        render_analysis(docs)
    if st.session_state.analysis:
        a = st.session_state.analysis
        st.subheader("Quick analysis")
        col1, col2 = st.columns(2)
        with col1:
            n_topics = st.slider("Number of LDA topics", 2, 10, 4)
            if st.button("Run LDA topic modeling"):
                result = run_lda([d["text"] for d in a["docs"]], n_topics=n_topics)
                st.session_state["lda_result"] = result
                st.success("LDA completed.")
        with col2:
            summary_mode = st.selectbox("Summarization method", ["TextRank (local)", "Transformer (optional)"])
            summary_len = st.selectbox("Summary size", ["Short", "Medium", "Detailed"], index=1)
            if st.button("Generate summary"):
                max_words = {"Short": 90, "Medium": 180, "Detailed": 300}[summary_len]
                if summary_mode.startswith("TextRank"):
                    summary = textrank_summary(a["combined"], max_words=max_words)
                else:
                    with st.spinner("Loading transformer model; first run may download model weights…"):
                        summary = transformer_summary(a["combined"], max_words=max_words)
                st.session_state["summary_result"] = summary
                if st.session_state.user:
                    save_analysis(st.session_state.user["id"], "summary", summary[:300], a["combined"][:50000])
                st.success("Summary generated.")
        if "lda_result" in st.session_state:
            st.markdown("### Topics discovered")
            topics = st.session_state["lda_result"]["topics"]
            for item in topics:
                st.write(f"**Topic {item['topic_id']}** — " + ", ".join(item["keywords"]))
            if topics:
                topic_df = pd.DataFrame([{"Topic": f"Topic {t['topic_id']}", "Keyword count": len(t["keywords"])} for t in topics])
                st.plotly_chart(px.bar(topic_df, x="Topic", y="Keyword count", title="Keywords per topic"), use_container_width=True)
        if "summary_result" in st.session_state:
            st.markdown("### Generated summary")
            st.write(st.session_state["summary_result"])
            st.download_button("Download summary (.txt)", st.session_state["summary_result"], file_name="summary.txt")
            pdf = build_report_pdf("NLP Insight Studio — Summary Report", st.session_state["summary_result"])
            st.download_button("Download summary report (.pdf)", pdf, file_name="nlp_summary_report.pdf", mime="application/pdf")

elif page == "Topic Modeling":
    st.subheader("Topic Modeling Lab")
    docs = load_inputs()
    if st.button("Run topic analysis", type="primary"):
        if not docs:
            st.error("Please add text or documents first.")
        else:
            n = st.slider("Number of LDA topics", 2, 12, 5, key="lda_n")
            result = run_lda([d["text"] for d in docs], n_topics=n)
            st.session_state["topic_page_result"] = result
            st.success("LDA topic modeling complete.")
    result = st.session_state.get("topic_page_result")
    if result:
        st.markdown("#### LDA topics")
        for t in result["topics"]:
            st.write(f"**Topic {t['topic_id']}** ({t['weight']:.2f} relative weight): " + ", ".join(t["keywords"]))
        st.metric("Topic coherence (c_v)", f"{result['coherence']:.3f}" if result["coherence"] is not None else "Not available")
        if result["topic_weights"]:
            df = pd.DataFrame(result["topic_weights"], columns=["Topic", "Weight"])
            st.plotly_chart(px.bar(df, x="Topic", y="Weight", title="Average topic distribution"), use_container_width=True)
        st.download_button("Download topics JSON", json.dumps(result, indent=2), file_name="topics.json")
    st.markdown("---")
    st.markdown("#### Optional BERTopic comparison")
    st.caption("BERTopic requires additional packages and may be slower; it is not required for the core workflow.")
    btxt = st.text_area("Text for BERTopic", height=130, key="bertopic_text")
    if st.button("Try BERTopic"):
        if not btxt.strip():
            st.error("Enter text first.")
        else:
            try:
                with st.spinner("Running BERTopic…"):
                    br = run_bertopic(btxt)
                st.json(br)
            except Exception as e:
                st.error(str(e))

elif page == "Summarization":
    st.subheader("Summarization Studio")
    docs = load_inputs()
    text_in = "\n\n".join(d["text"] for d in docs)
    if text_in:
        st.caption(f"Input length: {len(text_in.split()):,} words")
        method = st.radio("Method", ["Extractive — TextRank", "Abstractive — Transformer (optional)"], horizontal=True)
        length = st.select_slider("Target summary length", options=["Short", "Medium", "Detailed"], value="Medium")
        max_words = {"Short": 90, "Medium": 180, "Detailed": 300}[length]
        if st.button("Summarize", type="primary"):
            try:
                with st.spinner("Generating summary…"):
                    summary = textrank_summary(text_in, max_words) if method.startswith("Extractive") else transformer_summary(text_in, max_words)
                st.session_state["last_summary"] = summary
            except Exception as e:
                st.error(f"Summarization failed: {e}")
        if st.session_state.get("last_summary"):
            summary = st.session_state["last_summary"]
            left,right = st.columns(2)
            with left:
                st.markdown("**Original text**")
                st.text_area("Original", text_in[:10000], height=300, disabled=True)
            with right:
                st.markdown("**Summary**")
                st.text_area("Summary output", summary, height=300, disabled=True)
                st.download_button("Download summary", summary, file_name="summary.txt")
            stats = compression_stats(text_in, summary)
            a,b,c = st.columns(3)
            a.metric("Original words", stats["original_words"])
            b.metric("Summary words", stats["summary_words"])
            c.metric("Compression", f"{stats['compression_percent']:.1f}%")
    else:
        st.info("Upload or paste a document to begin.")

elif page == "Evaluation":
    st.subheader("Evaluation Workbench")
    original = st.text_area("Original/reference document", height=220)
    reference = st.text_area("Reference summary (human-written)", height=130)
    candidate = st.text_area("Generated summary to evaluate", height=130)
    if st.button("Calculate metrics", type="primary"):
        if not original.strip() or not candidate.strip():
            st.error("Enter an original document and candidate summary.")
        else:
            st.markdown("#### Length and compression")
            st.json(compression_stats(original, candidate))
            if reference.strip():
                st.markdown("#### ROUGE scores")
                st.json(rouge_scores(candidate, reference))
                st.caption("ROUGE measures lexical overlap; it does not by itself verify factual accuracy.")
            else:
                st.info("Add a human-written reference summary to calculate ROUGE scores.")
    st.markdown("#### What to measure")
    st.write("- **Topic coherence:** whether top words in each topic make semantic sense.")
    st.write("- **ROUGE-1/2/L:** overlap with a reference summary.")
    st.write("- **BERTScore (optional):** semantic similarity to a reference summary.")
    st.write("- **Human review:** factual consistency, coverage, readability, redundancy, and usefulness.")

elif page == "Multi-Document":
    st.subheader("Multi-Document Intelligence")
    docs = load_inputs()
    if docs:
        df = pd.DataFrame([{"Document": d["name"], "Words": len(d["text"].split()), "Characters": len(d["text"])} for d in docs])
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(px.bar(df, x="Document", y="Words", title="Document length comparison"), use_container_width=True)
        combined = "\n\n".join(d["text"] for d in docs)
        st.markdown("#### Combined summary")
        if st.button("Summarize all documents"):
            st.session_state["multi_summary"] = textrank_summary(combined, max_words=250)
        if st.session_state.get("multi_summary"):
            st.write(st.session_state["multi_summary"])
            st.download_button("Download combined summary", st.session_state["multi_summary"], file_name="combined_summary.txt")
        st.markdown("#### Shared topics")
        if st.button("Discover shared topics"):
            res = run_lda([d["text"] for d in docs], n_topics=min(6, max(2, len(docs))))
            for t in res["topics"]:
                st.write(f"**Topic {t['topic_id']}**: " + ", ".join(t["keywords"]))
    else:
        st.info("Upload two or more documents to compare them.")

elif page == "History":
    st.subheader("Analysis History")
    if not st.session_state.user:
        st.info("Sign in to view your saved history.")
    else:
        history = get_history(st.session_state.user["id"])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.info("No saved analyses yet.")

elif page == "Admin":
    st.subheader("Admin Console")
    user = st.session_state.user
    if not user or user["role"] != "admin":
        st.warning("Admin access required. Create an admin account using the setup instructions in README.md.")
    else:
        users = get_users()
        st.markdown("#### Registered users")
        st.dataframe(pd.DataFrame(users), use_container_width=True)
        st.metric("Registered users", len(users))
        st.caption("For production deployment, add stronger authorization controls, audit logging, and secure secret management.")

elif page == "About":
    st.subheader("About this project")
    st.write("NLP Insight Studio combines classical topic modeling, extractive summarization, optional transformer-based abstractive summarization, document comparison, and evaluation.")
    st.markdown("**Core methods:** LDA, TextRank, ROUGE, topic coherence.")
    st.markdown("**Optional extensions:** BERTopic, transformer summarization, BERTScore.")
    st.warning("Model output may omit important details or generate inaccurate claims. Always review summaries before using them for decisions.")
    st.markdown("**Suggested repository name:** `advanced-nlp-topic-modeling-summarization`")
