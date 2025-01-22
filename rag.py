# rag.py

import os
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import re

# If parser.py is in the same folder, make sure to reference it correctly
from parser import (
    extract_text_from_pdf,
    chunk_text_by_multiple_patterns,
    generate_regex_from_sample,
    concatenate_domain_control
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_embeddings(df_chunks, model_name='all-mpnet-base-v2'):
    logger.info("Generating embeddings for text chunks.")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df_chunks["Content"].tolist(), show_progress_bar=True)
    df_chunks["embedding"] = embeddings.tolist()
    logger.info("Embeddings generated.")
    return df_chunks, model

def create_faiss_index(df_chunks):
    logger.info("Creating FAISS index.")
    embedding_matrix = np.vstack(df_chunks["embedding"].values).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    logger.info("FAISS index created and embeddings added.")
    return index, embedding_matrix

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index saved to '{index_path}'.")

def load_faiss_index(index_path):
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    logger.info("FAISS index loaded successfully.")
    return index

def save_chunks_dataframe(df_chunks, df_chunks_path):
    logger.info(f"Saving chunks DataFrame to {df_chunks_path}")
    df_chunks_copy = df_chunks.copy()
    # Convert embedding arrays to comma-separated strings
    df_chunks_copy['embedding'] = df_chunks_copy['embedding'].apply(
        lambda x: ",".join(map(str, x))
    )
    df_chunks_copy.to_csv(df_chunks_path, index=False)
    logger.info(f"Chunks DataFrame saved to '{df_chunks_path}'.")

def load_chunks_dataframe(df_chunks_path):
    logger.info(f"Loading chunks DataFrame from {df_chunks_path}")
    df_chunks = pd.read_csv(df_chunks_path)
    # Convert comma-separated embedding strings back to float arrays
    df_chunks['embedding'] = df_chunks['embedding'].apply(
        lambda x: np.array([float(i) for i in x.split(",")])
    )
    logger.info("Chunks DataFrame loaded successfully.")
    return df_chunks

def embed_query(query, model):
    logger.debug(f"Embedding query: {query}")
    query_embedding = model.encode([query], show_progress_bar=False)
    return query_embedding.astype('float32')

def search_faiss(index, query_embedding, top_k=5):
    logger.debug(f"Searching FAISS index for top {top_k} results.")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def retrieve_answers(df_chunks, distances, indices):
    """
    Grab the matching chunks from df_chunks using the
    indices from FAISS, plus add a 'distance' column.
    """
    results = df_chunks.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    logger.debug("Retrieved answers from FAISS index search.")
    return results

def chunk_text_without_patterns(text, chunk_size=1000):
    """
    Splits the text into fixed-size chunks without using any patterns.
    """
    logger.info("Chunking text without patterns.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def build_rag_system_with_parser(
    pdf_path,
    start_page,
    end_page,
    control_patterns,
    output_text_path,
    df_chunks_path,
    faiss_index_path,
    chunk_size=1000
):
    """
    Extract text from a PDF, chunk it (with or without patterns),
    generate embeddings, and build the FAISS index.
    """
    logger.info("Building RAG system with parser.")
    text = extract_text_from_pdf(pdf_path, start_page, end_page, output_text_path)
    
    # Decide how to chunk
    if control_patterns:
        logger.info("Using pattern-based chunking for controls.")
        df_chunks = chunk_text_by_multiple_patterns(text, control_patterns)
    else:
        logger.info("Using fixed-size chunking for qualifiers.")
        chunks = chunk_text_without_patterns(text, chunk_size)
        df_chunks = pd.DataFrame({"Content": chunks})
    
    # Generate embeddings & save them
    df_chunks, model = generate_embeddings(df_chunks)
    save_chunks_dataframe(df_chunks, df_chunks_path)

    # Build the FAISS index & save
    index, _ = create_faiss_index(df_chunks)
    save_faiss_index(index, faiss_index_path)
    
    logger.info("RAG system built successfully with parser.")
    return model

def retrieve_answers_for_controls(
    df,
    model,
    index,
    df_chunks,
    top_k=3
):
    """
    For each control in df['Control'], retrieve the top_k matching chunks.
    Store them as Answer_1, Answer_2, ..., plus their corresponding Control ID.
    
    NOTE: We have removed the 'Page' field references to rely purely on
    'Control ID' and 'Content'.
    """
    logger.info("Retrieving answers for each control.")
    # Prepare the columns where answers will go
    for i in range(1, top_k + 1):
        df[f'Answer_{i}'] = None
        df[f'Answer_{i}_Control_ID'] = None

    # Iterate through each control in the DataFrame
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Controls"):
        query = row['Control']
        logger.debug(f"Processing Control: {query}")

        # 1) Embed the query
        query_emb = embed_query(query, model)

        # 2) Search the FAISS index
        distances, indices_search = search_faiss(index, query_emb, top_k)

        # 3) Retrieve results
        retrieved = retrieve_answers(df_chunks, distances, indices_search)
        
        # 4) Write the retrieved results to the DataFrame
        if retrieved.empty:
            logger.warning(f"No chunks retrieved for Control: {query}")
            continue
        
        for i in range(top_k):
            if i < len(retrieved):
                answer = retrieved.iloc[i]['Content']
                control_id = retrieved.iloc[i].get('Control ID', 'N/A')
                
                df.at[idx, f'Answer_{i+1}'] = answer
                df.at[idx, f'Answer_{i+1}_Control_ID'] = control_id

                logger.debug(
                    f"Retrieved Answer_{i+1} for Control '{query}': "
                    f"{answer[:50]}... Control_ID: {control_id}"
                )
            else:
                logger.debug(
                    f"No more answers available for Control '{query}' "
                    f"beyond Answer_{i+1}."
                )
    logger.info("Answers retrieved for all controls.")
    return df

def process_cybersecurity_framework_with_rag(
    excel_input_path,
    output_path,
    faiss_index_path,
    df_chunks_path,
    top_k=3
):
    """
    Load an Excel with 'Domain', 'Sub-Domain', 'Control' columns.
    Generate answers for each control, and save an updated file.
    """
    logger.info("Processing cybersecurity framework with RAG.")
    df = pd.read_excel(excel_input_path)
    required_columns = {'Domain', 'Sub-Domain', 'Control'}
    if not required_columns.issubset(df.columns):
        logger.error("Input Excel file missing required columns.")
        raise ValueError(f"Input Excel file must contain columns: {required_columns}")
    
    # This function concatenates 'Sub-Domain' and 'Control' -> 'Domain_Control'
    df = concatenate_domain_control(df)

    # Load model & index, plus chunk DataFrame
    model = SentenceTransformer('all-mpnet-base-v2')
    index = load_faiss_index(faiss_index_path)
    df_chunks = load_chunks_dataframe(df_chunks_path)

    # Retrieve answers
    df = retrieve_answers_for_controls(df, model, index, df_chunks, top_k=top_k)

    # Save updated DataFrame
    save_updated_framework(df, output_path)
    logger.info("Cybersecurity framework processed with RAG successfully.")
    return df

def save_updated_framework(df, output_path):
    """
    Save the final DataFrame (with answers) to Excel or CSV, 
    sanitizing non-ASCII characters if needed.
    """
    logger.info(f"Saving updated framework to {output_path}.")
    
    def sanitize_text(text):
        if isinstance(text, str):
            # Remove control characters, etc.
            return re.sub(r'[^ -\x7E]', ' ', text)
        return text

    df = df.applymap(sanitize_text)

    if output_path.lower().endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    logger.info(f"Updated DataFrame saved to '{output_path}'.")
