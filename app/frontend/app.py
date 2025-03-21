import time
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

# API configuration
API_URL = "http://localhost:8000/"  # Change to your GCP VM IP when deployed

# Set page configuration
st.set_page_config(
    page_title="NVIDIA Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .stTextInput input {
        border-radius: 5px;
    }
    .stSelectbox select {
        border-radius: 5px;
    }
    .quarter-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    .quarter-checkbox {
        padding: 5px;
        background-color: #f0f0f0;
        border-radius: 5px;
        text-align: center;
    }
    .answer-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin-top: 1rem;
        background-color: #f0f8ff;
    }
    .source-citation {
        background-color: #f1f1f1;
        border-left: 3px solid #4CAF50;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .info-box {
        background-color: #e1f5fe;
        border-left: 3px solid #03a9f4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .step-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .step-number {
        display: inline-block;
        width: 25px;
        height: 25px;
        background-color: #4CAF50;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 25px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "available_quarters" not in st.session_state:
    # Fetch available quarters from API
    try:
        response = requests.get(f"{API_URL}available_quarters")
        if response.status_code == 200:
            st.session_state.available_quarters = response.json()["quarters"]
        else:
            st.session_state.available_quarters = []
    except:
        # Default quarters if API is unavailable
        st.session_state.available_quarters = [
            "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", 
            "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
            "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4"
        ]

if "selected_quarters" not in st.session_state:
    st.session_state.selected_quarters = []
    
if "uploaded_document_id" not in st.session_state:
    st.session_state.uploaded_document_id = None

if "uploaded_document_name" not in st.session_state:
    st.session_state.uploaded_document_name = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = []

# Helper functions
def upload_pdf(file, extractor, chunking_method, vector_db, quarter=None):
    """Upload a PDF and process it with the selected options"""
    try:
        print("File: ", file)
        file_name = file.name
        files = {"file": (file_name, file.getvalue(), "application/pdf")}
        data = {
            "extractor": extractor,
            "chunking_method": chunking_method,
            "vector_db": vector_db
        }
        if quarter:
            data["quarter"] = quarter
            
        with st.spinner(f"Processing {file.name}..."):
            response = requests.post(
                f"{API_URL}process_pdf",
                files=files,
                data=data
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error processing PDF: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def ask_question(question, vector_db, quarter_filter=None, document_id=None, top_k=5):
    """Ask a question using the selected RAG options"""
    try:
        data = {
            "question": question,
            "vector_db": vector_db,
            "quarter_filter": quarter_filter,
            "top_k": top_k
        }
        if document_id:
            data["document_id"] = document_id
            
        with st.spinner("Searching knowledge base..."):
            response = requests.post(f"{API_URL}ask", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error asking question: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def toggle_quarter(quarter):
    """Toggle quarter selection"""
    if quarter in st.session_state.selected_quarters:
        st.session_state.selected_quarters.remove(quarter)
    else:
        st.session_state.selected_quarters.append(quarter)

# SIDEBAR
with st.sidebar:
    st.title("NVIDIA Financial RAG")
    st.markdown("---")
    
    # Vector DB Selection
    st.subheader("Vector Database")
    vector_db = st.radio(
        "Select vector database:",
        options=["ChromaDB", "Pinecone", "Embeddings.json"],
        index=0,
        key="vector_db"
    )
    
    st.markdown("---")
    
    # Quarter Selection - IMPROVED UI
    st.subheader("📅 Time Period Selection")
    
    # Add time period filter type
    filter_type = st.selectbox(
        "Filter by:",
        options=["Years & Quarters", "Specific Quarters", "Time Range"],
        index=0,
        help="Choose how you want to filter the time periods"
    )
    
    if filter_type == "Years & Quarters":
        # Group quarters by year with improved UI
        if st.session_state.available_quarters:
            # Extract years
            years = sorted(list(set(q.split("-")[0] for q in st.session_state.available_quarters)))
            
            # Year selection with multi-select
            selected_years = st.multiselect(
                "Select Years:",
                options=years,
                default=[years[-1]] if years else [],  # Default to latest year
                help="Select one or more years to filter data"
            )
            
            if selected_years:
                st.markdown("**Select Quarters:**")
                
                # Create a container with custom styling for quarters
                quarters_container = st.container()
                
                with quarters_container:
                    # Display quarters as colorful buttons
                    for year in selected_years:
                        st.markdown(f"""
                        <div style="background-color:#f0f8ff; padding:10px; border-radius:5px; margin-bottom:10px;">
                            <h4 style="margin:0; color:#333;">{year}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        year_quarters = [q for q in st.session_state.available_quarters if q.startswith(year)]
                        year_quarters.sort()
                        
                        # Quarter selection with button-like checkboxes
                        cols = st.columns(4)
                        for i, quarter in enumerate(year_quarters):
                            q_name = quarter.split("-")[1]
                            with cols[i % 4]:
                                quarter_selected = quarter in st.session_state.selected_quarters
                                
                                # Style based on selection state
                                if st.checkbox(
                                    q_name, 
                                    value=quarter_selected, 
                                    key=f"cb_{quarter}",
                                    help=f"Select {year} {q_name}"
                                ):
                                    if quarter not in st.session_state.selected_quarters:
                                        st.session_state.selected_quarters.append(quarter)
                                else:
                                    if quarter in st.session_state.selected_quarters:
                                        st.session_state.selected_quarters.remove(quarter)
                
                # Quick selection buttons for current year
                if selected_years and len(selected_years) == 1:
                    st.markdown("**Quick Select:**")
                    quick_cols = st.columns(3)
                    year = selected_years[0]
                    year_quarters = sorted([q for q in st.session_state.available_quarters if q.startswith(year)])
                    
                    with quick_cols[0]:
                        if st.button(f"All {year}", key=f"all_{year}"):
                            for q in year_quarters:
                                if q not in st.session_state.selected_quarters:
                                    st.session_state.selected_quarters.append(q)
                    
                    with quick_cols[1]:
                        if st.button(f"First Half {year}", key=f"h1_{year}"):
                            half1 = [q for q in year_quarters if q.endswith("Q1") or q.endswith("Q2")]
                            for q in half1:
                                if q not in st.session_state.selected_quarters:
                                    st.session_state.selected_quarters.append(q)
                    
                    with quick_cols[2]:
                        if st.button(f"Second Half {year}", key=f"h2_{year}"):
                            half2 = [q for q in year_quarters if q.endswith("Q3") or q.endswith("Q4")]
                            for q in half2:
                                if q not in st.session_state.selected_quarters:
                                    st.session_state.selected_quarters.append(q)
    
    elif filter_type == "Specific Quarters":
        # Better UI for quarter selection with color coding
        st.markdown("""
        <style>
        div.q1-item {background-color: #e1f5fe; padding: 5px; border-radius: 5px; margin: 2px 0;}
        div.q2-item {background-color: #e8f5e9; padding: 5px; border-radius: 5px; margin: 2px 0;}
        div.q3-item {background-color: #fff3e0; padding: 5px; border-radius: 5px; margin: 2px 0;}
        div.q4-item {background-color: #f3e5f5; padding: 5px; border-radius: 5px; margin: 2px 0;}
        </style>
        """, unsafe_allow_html=True)
        
        # Organize quarters by Q1, Q2, Q3, Q4
        all_quarters = st.session_state.available_quarters
        q1_quarters = sorted([q for q in all_quarters if q.endswith("Q1")])
        q2_quarters = sorted([q for q in all_quarters if q.endswith("Q2")])
        q3_quarters = sorted([q for q in all_quarters if q.endswith("Q3")])
        q4_quarters = sorted([q for q in all_quarters if q.endswith("Q4")])
        
        # Create tabs for different quarters
        q_tabs = st.tabs(["Q1", "Q2", "Q3", "Q4"])
        
        # Q1 Tab
        with q_tabs[0]:
            st.markdown('<div style="text-align:center; color:#0277bd; font-weight:bold;">First Quarter Selection</div>', unsafe_allow_html=True)
            for q in q1_quarters:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="q1-item">{q}</div>', unsafe_allow_html=True)
                with col2:
                    if st.checkbox("", value=q in st.session_state.selected_quarters, key=f"tab_cb_{q}"):
                        if q not in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.append(q)
                    else:
                        if q in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.remove(q)
        
        # Q2 Tab
        with q_tabs[1]:
            st.markdown('<div style="text-align:center; color:#2e7d32; font-weight:bold;">Second Quarter Selection</div>', unsafe_allow_html=True)
            for q in q2_quarters:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="q2-item">{q}</div>', unsafe_allow_html=True)
                with col2:
                    if st.checkbox("", value=q in st.session_state.selected_quarters, key=f"tab_cb_{q}"):
                        if q not in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.append(q)
                    else:
                        if q in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.remove(q)
        
        # Q3 Tab
        with q_tabs[2]:
            st.markdown('<div style="text-align:center; color:#ef6c00; font-weight:bold;">Third Quarter Selection</div>', unsafe_allow_html=True)
            for q in q3_quarters:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="q3-item">{q}</div>', unsafe_allow_html=True)
                with col2:
                    if st.checkbox("", value=q in st.session_state.selected_quarters, key=f"tab_cb_{q}"):
                        if q not in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.append(q)
                    else:
                        if q in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.remove(q)
        
        # Q4 Tab
        with q_tabs[3]:
            st.markdown('<div style="text-align:center; color:#7b1fa2; font-weight:bold;">Fourth Quarter Selection</div>', unsafe_allow_html=True)
            for q in q4_quarters:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="q4-item">{q}</div>', unsafe_allow_html=True)
                with col2:
                    if st.checkbox("", value=q in st.session_state.selected_quarters, key=f"tab_cb_{q}"):
                        if q not in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.append(q)
                    else:
                        if q in st.session_state.selected_quarters:
                            st.session_state.selected_quarters.remove(q)
    
    elif filter_type == "Time Range":
        # Time range selector
        if st.session_state.available_quarters:
            # Get min and max years
            all_years = sorted(list(set(q.split("-")[0] for q in st.session_state.available_quarters)))
            min_year = all_years[0]
            max_year = all_years[-1]
            
            st.markdown("**Select Range:**")
            
            # Select start period
            start_col, end_col = st.columns(2)
            
            with start_col:
                start_year = st.selectbox("From Year:", all_years, index=0, key="range_start_year")
                start_quarters = ["Q1", "Q2", "Q3", "Q4"]
                start_q = st.selectbox("From Quarter:", start_quarters, index=0, key="range_start_q")
            
            with end_col:
                end_year = st.selectbox("To Year:", all_years, index=len(all_years)-1, key="range_end_year")
                end_quarters = ["Q1", "Q2", "Q3", "Q4"]
                end_q = st.selectbox("To Quarter:", end_quarters, index=3, key="range_end_q")
            
            # Apply range button with nicer styling
            if st.button("Apply Range", type="primary", key="apply_range"):
                # Clear previous selection
                st.session_state.selected_quarters = []
                
                # Calculate start and end period indices
                start_index = (int(start_year) - int(min_year)) * 4 + start_quarters.index(start_q)
                end_index = (int(end_year) - int(min_year)) * 4 + end_quarters.index(end_q)
                
                # Add all quarters in the selected range
                for year in all_years:
                    for q in ["Q1", "Q2", "Q3", "Q4"]:
                        period = f"{year}-{q}"
                        current_index = (int(year) - int(min_year)) * 4 + ["Q1", "Q2", "Q3", "Q4"].index(q)
                        
                        if start_index <= current_index <= end_index and period in st.session_state.available_quarters:
                            st.session_state.selected_quarters.append(period)
    
    # Add a visual indicator of selected quarters
    if st.session_state.selected_quarters:
        sorted_quarters = sorted(st.session_state.selected_quarters)
        
        # Create a styled container to show selected quarters
        st.markdown("""
        <style>
        .selected-quarters {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        .quarter-tag {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div class='selected-quarters'><b>Selected ({len(sorted_quarters)}):</b><br>" + 
                    "".join([f"<span class='quarter-tag'>{q}</span>" for q in sorted_quarters]) +
                    "</div>", unsafe_allow_html=True)
        
        # Clear Selection button
        if st.button("Clear Selection", type="secondary"):
            st.session_state.selected_quarters = []
            st.rerun()
    else:
        st.info("No quarters selected. All data will be searched.")

# MAIN CONTENT AREA
st.title("NVIDIA Financial RAG System")

# Create tabs
tab1, tab2 = st.tabs(["NVIDIA Question Answering", "Upload & Query PDF"])

# Tab 1: NVIDIA Question Answering
with tab1:
    st.header("Ask Questions About NVIDIA Financials")
    
    # Instructions
    with st.expander("How to Use This System", expanded=False):
        st.markdown("""
        <div class="info-box">
            <h4>How to use the NVIDIA Financial Question Answering System:</h4>
            
            <div class="step-container">
                <span class="step-number">1</span> <strong>Select Vector Database</strong>: Choose from ChromaDB, Pinecone, or Embeddings.json in the sidebar.
            </div>
            
            <div class="step-container">
                <span class="step-number">2</span> <strong>Select Quarters</strong>: Choose which financial quarters to include in your search. 
                You can select specific quarters or use "Select All"/"Clear All" buttons.
            </div>
            
            <div class="step-container">
                <span class="step-number">3</span> <strong>Ask Your Question</strong>: Type a question about NVIDIA financials in the text box below.
            </div>
            
            <div class="step-container">
                <span class="step-number">4</span> <strong>View Results</strong>: The system will retrieve relevant information and generate an answer using Google's Gemini model.
            </div>
            
            <p>Example questions:</p>
            <ul>
                <li>What was NVIDIA's revenue in Q1 2022?</li>
                <li>How did NVIDIA's gaming segment perform in 2021?</li>
                <li>What are the main drivers of NVIDIA's growth?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Question input
    nvidia_question = st.text_area("Ask a question about NVIDIA financials:", 
                              placeholder="e.g., What was NVIDIA's revenue growth in 2021?", 
                              height=100,
                              key="nvidia_question")
    
    # Get answer button with selected options display
    col1, col2 = st.columns([1, 3])
    with col1:
        nvidia_get_answer = st.button("Get Answer", type="primary", key="nvidia_get_answer")
    
    with col2:
        quarters_text = ", ".join(sorted(st.session_state.selected_quarters)) if st.session_state.selected_quarters else "All quarters"
        st.caption(f"Using {vector_db} | Quarters: {quarters_text}")
    
    # Process question if button clicked
    if nvidia_get_answer and nvidia_question:
        response = ask_question(
            nvidia_question,
            vector_db.lower().replace(".", ""),
            quarter_filter=st.session_state.selected_quarters if st.session_state.selected_quarters else None
        )
        
        if response:
            st.session_state.answer = response.get("answer", "")
            st.session_state.context_chunks = response.get("context_chunks", [])
            st.session_state.processing_time = response.get("processing_time", 0)
            st.session_state.token_info = response.get("token_info", None)
    
    # Display answer and sources
    if hasattr(st.session_state, 'answer') and st.session_state.answer:
        st.markdown("## Answer")
        st.markdown(st.session_state.answer)
        
        if hasattr(st.session_state, 'processing_time'):
            st.caption(f"Processing time: {st.session_state.processing_time:.2f} seconds")
        
        if hasattr(st.session_state, 'token_info') and st.session_state.token_info:
            token_info = st.session_state.token_info
            
            with st.expander("Token Usage and Cost Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Input Tokens", f"{token_info['input_tokens']:,}")
                    st.metric("Output Tokens", f"{token_info['output_tokens']:,}")
                    st.metric("Total Tokens", f"{token_info['total_tokens']:,}")
                with col2:
                    st.metric("Input Cost", f"${token_info['input_cost']:.5f}")
                    st.metric("Output Cost", f"${token_info['output_cost']:.5f}")
                    st.metric("Total Cost", f"${token_info['total_cost']:.5f}")
                st.caption(f"Model: {token_info['model']}")
                
        # Show source documents
        with st.expander("View Source Documents", expanded=False):
            if st.session_state.context_chunks:
                for i, chunk in enumerate(st.session_state.context_chunks):
                    st.markdown(f"**Source {i+1}:** {chunk.get('document_id', 'Unknown')} - {chunk.get('quarter', 'Unknown')} (Score: {chunk.get('similarity', 0):.3f})")
                    st.markdown(f"<div class='source-citation'>{chunk.get('text', '')}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                else:
                    st.info("No specific source documents were used for this answer.")

# Tab 2: Upload & Query PDF
with tab2:
    st.header("Upload & Query Your PDF")
    
    # Instructions
    with st.expander("How to Upload and Query PDFs", expanded=False):
        st.markdown("""
        <div class="info-box">
            <h4>How to Upload and Query Your Own PDFs:</h4>
            
            <div class="step-container">
                <span class="step-number">2</span> <strong>Configure Processing Options</strong>: 
                <ul>
                    <li><strong>Text Extractor</strong>: Choose between Docling (standard) or Mistral OCR (better for scanned documents)</li>
                    <li><strong>Chunking Method</strong>: Select how to divide the document (Fixed, Markdown, or Sentence-based)</li>
                    <li><strong>Vector Database</strong>: Choose where to store the processed document</li>
                </ul>
            </div>
            
            <div class="step-container">
                <span class="step-number">3</span> <strong>Process the PDF</strong>: Click the "Process PDF" button to extract, chunk, and store the document.
            </div>
            
            <div class="step-container">
                <span class="step-number">4</span> <strong>Ask Questions</strong>: Once processed, you can ask questions about your document.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # PDF Upload Section
    st.subheader("Upload PDF Document")
    
    # Show the upload interface only if no document is already uploaded
    if not st.session_state.uploaded_document_id:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configure Processing Options:**")
                
                # Text extractor selection
                text_extractor = st.radio(
                    "Text Extractor",
                    options=["Docling", "Mistral OCR"],
                    index=0,
                    help="Docling is standard, Mistral OCR is better for scanned documents"
                )
                
                # Chunking method selection
                chunking_method = st.radio(
                    "Chunking Method",
                    options=["Fixed", "Markdown", "Sentence"],
                    index=0,
                    help="How to split the document into chunks"
                )
                
                # Quarter assignment (optional)
                pdf_quarter = st.selectbox(
                    "Assign Quarter (optional)",
                    options=["None"] + sorted(st.session_state.available_quarters),
                    index=0,
                    help="Assign a quarter to this document for filtering"
                )
                
            with col2:
                st.write(f"**Selected File:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                
                # Vector DB selection for this upload
                pdf_vector_db = st.radio(
                    "Vector Database",
                    options=["ChromaDB", "Pinecone", "Embeddings.json"],
                    index=0,
                    help="Where to store the document embeddings"
                )
                
                # Process button
                if st.button("Process PDF", type="primary"):
                    # Convert UI selections to expected API parameters
                    extractor_param = text_extractor.lower().replace(" ", "_")
                    chunking_param = chunking_method.lower()
                    vector_db_param = pdf_vector_db.lower().replace(".", "")
                    quarter_param = None if pdf_quarter == "None" else pdf_quarter
                    
                    # Call the API to process the PDF
                    result = upload_pdf(
                        uploaded_file,
                        extractor_param,
                        chunking_param,
                        vector_db_param,
                        quarter_param
                    )
                    
                    if result:
                        st.success(f"PDF uploaded and processed successfully!")
                        # Store the document ID for future reference
                        st.session_state.uploaded_document_id = result["document_id"]
                        st.session_state.uploaded_document_name = uploaded_file.name
                        st.session_state.uploaded_vector_db = vector_db_param
                        
                        # Display processing details
                        st.write(f"Created {result['chunks']} chunks")
                        st.write(f"Stored in {result['vector_db']}")
                        
                        # Rerun to update the UI
                        st.rerun()
    
    # If a document is uploaded, show the query interface
    else:
        # Document info and reset button
        st.info(f"Current document: **{st.session_state.uploaded_document_name}** (ID: {st.session_state.uploaded_document_id[:8]}...)")
        
        if st.button("Reset (Upload New Document)"):
            st.session_state.uploaded_document_id = None
            st.session_state.uploaded_document_name = None
            st.session_state.answer = None
            st.session_state.context_chunks = []
            st.rerun()
        
        # Only show the ask questions section after a document has been processed
        st.subheader("Ask Questions About Your Document")
        
        # Question input
        pdf_question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main points in this document?",
            height=100,
            key="pdf_question"
        )
        
        # Get answer button
        if st.button("Get Answer", type="primary", key="pdf_get_answer"):
            if pdf_question:
                # Use the document_id to query only this document
                response = ask_question(
                    pdf_question,
                    st.session_state.uploaded_vector_db,
                    document_id=st.session_state.uploaded_document_id
                )
                
                if response:
                    st.session_state.answer = response.get("answer", "")
                    st.session_state.context_chunks = response.get("context_chunks", [])
                    st.session_state.processing_time = response.get("processing_time", 0)
                    st.session_state.token_info = response.get("token_info", None)
        else:
                st.warning("Please enter a question.")
        
        # Display answer and sources
        if hasattr(st.session_state, 'answer') and st.session_state.answer:
            st.markdown("## Answer")
            st.markdown(st.session_state.answer)
            
            if hasattr(st.session_state, 'processing_time'):
                st.caption(f"Processing time: {st.session_state.processing_time:.2f} seconds")
            
            if hasattr(st.session_state, 'token_info') and st.session_state.token_info:
                token_info = st.session_state.token_info
                
                with st.expander("Token Usage and Cost Information", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Tokens", f"{token_info['input_tokens']:,}")
                        st.metric("Output Tokens", f"{token_info['output_tokens']:,}")
                        st.metric("Total Tokens", f"{token_info['total_tokens']:,}")
                    with col2:
                        st.metric("Input Cost", f"${token_info['input_cost']:.5f}")
                        st.metric("Output Cost", f"${token_info['output_cost']:.5f}")
                        st.metric("Total Cost", f"${token_info['total_cost']:.5f}")
                    st.caption(f"Model: {token_info['model']}")
                    
            # Show source documents
            with st.expander("View Source Chunks", expanded=False):
                if st.session_state.context_chunks:
                    for i, chunk in enumerate(st.session_state.context_chunks):
                        st.markdown(f"**Chunk {i+1}** (Score: {chunk.get('similarity', 0):.3f})")
                        st.markdown(f"<div class='source-citation'>{chunk.get('text', '')}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.info("No specific source chunks were used for this answer.")

# Footer
st.markdown("---")
st.caption("NVIDIA Financial RAG System | Powered by Google Gemini")