import os
from datetime import datetime
import json

def process_sample_data():
    """
    Process a sample data file for demonstration purposes.
    """
    # Create sample data
    sample_data = [
        {
            "title": "Artificial Intelligence",
            "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Machine Learning",
            "content": "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Deep Learning",
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "url": "https://en.wikipedia.org/wiki/Deep_learning",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
    
    # Write sample data to file
    output_file = os.path.join(os.getcwd(), "data", "sample_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data created at {output_file}")
    print(f"Generated {len(sample_data)} sample records")
    
    # Return file path for further processing
    return output_file


def clean_sample_data(input_file, output_file=None):
    """
    Clean sample data for demonstration purposes.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional)
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    # Read data from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Perform cleaning operations
    for item in data:
        # Trim whitespace
        for key in item:
            if isinstance(item[key], str):
                item[key] = item[key].strip()
        
        # Add additional metadata
        item['cleaned'] = True
        item['cleaned_at'] = datetime.now().isoformat()
    
    # Write cleaned data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned data saved to {output_file}")
    print(f"Cleaned {len(data)} records")
    
    return output_file


if __name__ == "__main__":
    # Process sample data
    input_file = process_sample_data()
    
    # Clean sample data
    clean_sample_data(input_file)
