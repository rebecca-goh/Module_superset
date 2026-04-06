"""
Process TSF file of a product to compare with other product with ML 
1. Sentence Transformers - Semantic similarity using embeddings (best for semantic meaning)
2. TF-IDF + Cosine Similarity - Traditional text similarity (good for keyword matching)

The script automatically searches in the first column that starts with "Parameter" in the CSV file.

Usage:
    python run_speccomparison.py "search string" path/to/file.csv [options]

Requirements:
    pip install -r requirements_ml_matcher.txt
    
    Or individually:
    pip install pandas scikit-learn rapidfuzz sentence-transformers numpy
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ML method imports (with fallbacks)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def find_parameter_column(df):
    """Find the first column that starts with 'TestParameter'."""
    for col in df.columns:
        if str(col).startswith('TestParameter'):
            return col
    raise ValueError("No column starting with 'TestParameter found in the CSV file")


def find_test_number_row(df):
    """Find the row index where the first cell starts with 'TestNumber'.
    
    Args:
        df: pandas DataFrame to search
        
    Returns:
        int: Row index (0-based) where 'TestNumber' is found
        
    Raises:
        ValueError: If no row starting with 'TestNumber' is found
    """
    for idx, row in df.iterrows():
        # Check the first cell (first column) of each row
        first_cell = str(row.iloc[0])
        if first_cell.startswith('TestNumber'):
            return idx
    
    raise ValueError("No row starting with 'TestNumber' found in the CSV file")

def method_sentence_transformers(search_string, df, column, top_n=1):
    """Find closest matches using sentence transformers (semantic similarity)."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers library is not installed. Install it with: pip install sentence-transformers")
    
    # Load model (using a lightweight model)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get all strings from the column
    strings = df[column].astype(str).tolist()
    
    # Create embeddings
    search_embedding = model.encode([search_string])
    string_embeddings = model.encode(strings)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(search_embedding, string_embeddings)[0]
    
    # Get top N matches
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': df.index[idx],
            'match': strings[idx],
            'similarity': float(similarities[idx]),
            'row_data': df.iloc[idx].to_dict()
        })
    
    return results


def method_tfidf(search_string, df, column, top_n=1):
    """Find closest matches using TF-IDF and cosine similarity."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn library is not installed. Install it with: pip install scikit-learn")
    
    # Get all strings from the column
    strings = df[column].astype(str).tolist()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform all strings including search string
    all_texts = [search_string] + strings
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between search string and all strings
    search_vector = tfidf_matrix[0:1]
    string_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(search_vector, string_vectors)[0]
    
    # Get top N matches
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': df.index[idx],
            'match': strings[idx],
            'similarity': float(similarities[idx]),
            'row_data': df.iloc[idx].to_dict()
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Find the closest matching string in a CSV file using machine learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_closest_string.py "temperature sensor" data.csv --method tfidf
  python find_closest_string.py "voltage" data.csv --method fuzzy --top-n 5
  python find_closest_string.py "pressure" data.csv --method sentence-transformers
        """
    )
    
    parser.add_argument('--source_csv', type=str, help='Input string to search for')
    parser.add_argument('--compare_list', type=str, help='List of Path to the CSV file')
    parser.add_argument('--method', type=str, 
                       choices=['sentence-transformers', 'tfidf'],
                       default='sentence-transformers',
                       help='ML method to use (default: tfidf)')
    args = parser.parse_args()
    
    # Validate CSV file exists
    source_csv = Path(args.source_csv)
    compare_paths = args.compare_list
    if not source_csv.exists():
        print(f"Error: CSV file not found: {args.source_csv}", file=sys.stderr)
        sys.exit(1)
    
    # Load CSV file
    try:
        df_source = pd.read_csv(args.source_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Find Parameter row
    try:
        param_row= find_test_number_row(df_source)
        print(f"Found row: '{param_row}'")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract parameters from source CSV (columns B, F, G)
    # Column B = index 1, Column F = index 5, Column G = index 6
    source_parameters = []
    
    print(f"\nExtracting parameters from source CSV starting at row {param_row}...")
    for idx in range(param_row, len(df_source)):
        row = df_source.iloc[idx]
        
        # Column B (index 1) - typically Parameter name
        col_b = str(row.iloc[1]) if len(row) > 1 else ''
        
        # Column F (index 5) - typically Min spec
        col_f = str(row.iloc[5]) if len(row) > 5 else ''
        
        # Column G (index 6) - typically Max spec
        col_g = str(row.iloc[6]) if len(row) > 6 else ''
        
        # Skip empty rows
        if col_b.strip() and col_b.lower() not in ['nan', 'none', '']:
            parameter_data = {
                'row_index': idx,
                'parameter_name': col_b,
                'column_f': col_f,
                'column_g': col_g
            }
            source_parameters.append(parameter_data)
            print(f"  Row {idx}: Parameter='{col_b}', Col_F='{col_f}', Col_G='{col_g}'")
    
    # Convert to DataFrame
    df_source_parameters = pd.DataFrame(source_parameters)
    df_source_parameters.to_csv('source_parameters.csv', index=False)
    print(f"\nSource Parameters DataFrame:")
    print(df_source_parameters.to_string(index=False))
    
    # Load compare file to find matching parameter and spec values
    # Split the comma-separated string and strip whitespace
    print(compare_paths)
    compare_path_lists = compare_paths.split(',')
    
    all_matches = []  # Store all matching results
    
    for compare_path in compare_path_lists:
        if not compare_path:  # Skip empty strings
            continue
            
        csv_path = Path(compare_path)
        
        # Check if file exists
        if not csv_path.exists():
            print(f"Error: File not found: {compare_path}", file=sys.stderr)
            continue
        
        # Try to load the CSV file
        try:
            df_compare = pd.read_csv(csv_path)
            print(f"\n{'='*80}")
            print(f"Successfully loaded: {csv_path.name} ({len(df_compare)} rows, {len(df_compare.columns)} columns)")
        except Exception as e:
            print(f"Failed to load {csv_path.name}: {str(e)}", file=sys.stderr)
            continue
        
        # Find parameter row in compare file
        try:
            compare_param_row = find_test_number_row(df_compare)
            print(f"Found parameter row in compare file at index: {compare_param_row}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        
        # Extract parameters from compare CSV (column B)
        compare_parameters = []
        for idx in range(compare_param_row, len(df_compare)):
            row = df_compare.iloc[idx]
            col_b = str(row.iloc[1]) if len(row) > 1 else ''
            col_f = str(row.iloc[5]) if len(row) > 5 else ''
            col_g = str(row.iloc[6]) if len(row) > 6 else ''
            
            if col_b.strip() and col_b.lower() not in ['nan', 'none', '']:
                compare_parameters.append({
                    'row_index': idx,
                    'parameter_name': col_b,
                    'column_f': col_f,
                    'column_g': col_g
                })
        
        df_compare_parameters = pd.DataFrame(compare_parameters)
        print(f"Extracted {len(df_compare_parameters)} parameters from compare file")
        
        # Match each source parameter with compare parameters
        print(f"\n{'='*80}")
        print(f"Matching parameters using method: {args.method}")
        print(f"{'='*80}\n")
        
        for _, source_row in df_source_parameters.iterrows():
            source_param = source_row['parameter_name']
            
            # Find closest match using ML method
            try:
                if args.method == 'sentence-transformers':
                    matches = method_sentence_transformers(
                        source_param, 
                        df_compare_parameters, 
                        'parameter_name', 
                        top_n=1
                    )
                elif args.method == 'tfidf':
                    matches = method_tfidf(
                        source_param, 
                        df_compare_parameters, 
                        'parameter_name', 
                        top_n=1
                    )
                
                if matches:
                    best_match = matches[0]
                    match_result = {
                        'source_file': args.source_csv,
                        'compare_file': csv_path.name,
                        'source_param': source_param,
                        'source_min': source_row['column_f'],
                        'source_max': source_row['column_g'],
                        'matched_param': best_match['match'],
                        'matched_min': best_match['row_data']['column_f'],
                        'matched_max': best_match['row_data']['column_g'],
                        'similarity_score': best_match['similarity'],
                        'source_row_index': source_row['row_index'],
                        'compare_row_index': best_match['row_data']['row_index']
                    }
                    all_matches.append(match_result)
                    
                    print(f"Source: '{source_param}'")
                    print(f"  -> Matched: '{best_match['match']}' (Score: {best_match['similarity']:.4f})")
                    print(f"  -> Source [F={source_row['column_f']}, G={source_row['column_g']}]")
                    print(f"  -> Compare [F={best_match['row_data']['column_f']}, G={best_match['row_data']['column_g']}]")
                    print()
                    
            except Exception as e:
                print(f"Error matching '{source_param}': {e}", file=sys.stderr)
                continue
    
    # Convert all matches to DataFrame and save
    if all_matches:
        df_all_matches = pd.DataFrame(all_matches)
        output_file = 'parameter_matches.csv'
        df_all_matches.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"All matches saved to: {output_file}")
        print(f"Total matches found: {len(df_all_matches)}")
        print(f"\nMatching Results Summary:")
        print(df_all_matches.to_string(index=False))
    else:
        print("\nNo matches found.")


if __name__ == '__main__':
    main()

