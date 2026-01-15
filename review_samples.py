#!/usr/bin/env python3
"""
Quick manual review of filtered datasets
Extracts random samples and displays side-by-side for human evaluation
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def extract_samples(input_file: str, num_samples: int = 200, seed: int = 42):
    """Extract random samples from dataset"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Shuffle and sample
    random.seed(seed)
    samples = random.sample(data, min(num_samples, len(data)))
    
    return samples

def display_samples(samples: list, title: str = "Samples"):
    """Display samples in readable format"""
    
    print("\n" + "="*80)
    print(f"{title} ({len(samples)} pairs)")
    print("="*80)
    
    for i, item in enumerate(samples, 1):
        ko = item['translation']['kor_Hang']
        vi = item['translation']['vie_Latn']
        
        print(f"\n[{i}]")
        print(f"KO: {ko}")
        print(f"VI: {vi}")
        
        # Show lengths
        ko_len = len(ko.split())
        vi_len = len(vi.split())
        ratio = vi_len / ko_len if ko_len > 0 else 0
        
        print(f"    Lengths: KO={ko_len} words, VI={vi_len} words, Ratio={ratio:.2f}")
        
        # Check for numbers
        import re
        ko_nums = re.findall(r'\d+', ko)
        vi_nums = re.findall(r'\d+', vi)
        if ko_nums or vi_nums:
            print(f"    Numbers: KO={ko_nums}, VI={vi_nums}")
        
        if i % 10 == 0:
            print("\n" + "-"*80)

def save_samples_html(samples: list, output_file: str, title: str = "Sample Review"):
    """Save samples in HTML format for easier review"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .pair {{
            background: white;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .pair-number {{
            font-weight: bold;
            color: #666;
            margin-bottom: 10px;
        }}
        .korean {{
            margin: 10px 0;
            padding: 10px;
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
        }}
        .vietnamese {{
            margin: 10px 0;
            padding: 10px;
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }}
        .lang-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
        }}
        .text {{
            font-size: 1.1em;
            margin-top: 5px;
        }}
        .stats {{
            margin-top: 10px;
            padding: 10px;
            background-color: #fff3e0;
            border-left: 4px solid #FF9800;
            font-size: 0.9em;
            color: #666;
        }}
        .quality-check {{
            margin-top: 15px;
            padding: 10px;
            background-color: #fce4ec;
            border-radius: 4px;
        }}
        .quality-check label {{
            margin-right: 15px;
            cursor: pointer;
        }}
        .quality-check input {{
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p style="text-align: center; color: #666;">Total pairs: {len(samples)}</p>
"""
    
    for i, item in enumerate(samples, 1):
        ko = item['translation']['kor_Hang']
        vi = item['translation']['vie_Latn']
        
        ko_len = len(ko.split())
        vi_len = len(vi.split())
        ratio = vi_len / ko_len if ko_len > 0 else 0
        
        import re
        ko_nums = re.findall(r'\d+', ko)
        vi_nums = re.findall(r'\d+', vi)
        
        html += f"""
    <div class="pair">
        <div class="pair-number">Pair #{i}</div>
        
        <div class="korean">
            <div class="lang-label">🇰🇷 Korean:</div>
            <div class="text">{ko}</div>
        </div>
        
        <div class="vietnamese">
            <div class="lang-label">🇻🇳 Vietnamese:</div>
            <div class="text">{vi}</div>
        </div>
        
        <div class="stats">
            📊 <strong>Lengths:</strong> KO={ko_len} words, VI={vi_len} words, Ratio={ratio:.2f}
            {f' | 🔢 <strong>Numbers:</strong> KO={ko_nums}, VI={vi_nums}' if ko_nums or vi_nums else ''}
        </div>
        
        <div class="quality-check">
            <strong>Quality Rating:</strong>
            <label><input type="radio" name="quality_{i}" value="good"> ✅ Good alignment</label>
            <label><input type="radio" name="quality_{i}" value="style"> 🟡 Different style</label>
            <label><input type="radio" name="quality_{i}" value="wrong"> ❌ Wrong translation</label>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def compare_thresholds(threshold_files: dict, num_samples: int = 50):
    """Compare quality across different thresholds"""
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    
    all_samples = {}
    
    for threshold, file_path in threshold_files.items():
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        samples = extract_samples(file_path, num_samples)
        all_samples[threshold] = samples
        
        print(f"\n{threshold:.2f}: {len(samples)} samples extracted from {file_path}")
    
    # Save comparison HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Threshold Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 { text-align: center; }
        .threshold-section {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .threshold-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .pair {
            margin: 15px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .ko { color: #1976D2; font-weight: 500; }
        .vi { color: #388E3C; font-weight: 500; }
    </style>
</head>
<body>
    <h1>🔍 Semantic Threshold Comparison</h1>
"""
    
    for threshold in sorted(all_samples.keys()):
        samples = all_samples[threshold]
        html += f"""
    <div class="threshold-section">
        <div class="threshold-title">Threshold: {threshold:.2f} ({len(samples)} samples)</div>
"""
        
        for i, item in enumerate(samples[:10], 1):  # Show first 10
            ko = item['translation']['kor_Hang']
            vi = item['translation']['vie_Latn']
            
            html += f"""
        <div class="pair">
            <div><strong>[{i}]</strong></div>
            <div class="ko">KO: {ko}</div>
            <div class="vi">VI: {vi}</div>
        </div>
"""
        
        html += "    </div>\n"
    
    html += "</body></html>"
    
    output_file = "threshold_comparison.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Comparison saved to: {output_file}")
    print("   Open in browser to review side-by-side")

def main():
    parser = argparse.ArgumentParser(description="Manual review tool for filtered datasets")
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples to extract')
    parser.add_argument('--output-html', type=str, help='Output HTML file for review')
    parser.add_argument('--compare', action='store_true', help='Compare multiple thresholds')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.65, 0.70, 0.75, 0.80],
                       help='Thresholds to compare')
    parser.add_argument('--filtered-dir', type=str, default='data/sweep_filtered',
                       help='Directory with semantic_XX.jsonl files')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        threshold_files = {}
        for t in args.thresholds:
            threshold_int = int(t * 100)
            file_path = f"{args.filtered_dir}/semantic_{threshold_int}.jsonl"
            threshold_files[t] = file_path
        
        compare_thresholds(threshold_files, args.num_samples)
        
    elif args.input:
        # Single file mode
        samples = extract_samples(args.input, args.num_samples)
        
        # Display in terminal
        display_samples(samples, f"Samples from {args.input}")
        
        # Save HTML if requested
        if args.output_html:
            save_samples_html(samples, args.output_html, f"Review: {args.input}")
            print(f"\n✅ HTML saved to: {args.output_html}")
            print("   Open in browser for easier review")
    
    else:
        parser.error("Either --input or --compare must be specified")

if __name__ == "__main__":
    main()
