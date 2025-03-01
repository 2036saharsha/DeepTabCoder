import json
import subprocess
import os
import argparse

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_file(file_name, content, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    with open(path, 'w') as file:
        file.write(content)

def execute_python_file(file_name, folder):
    try:
        result = subprocess.run(
            ['python', os.path.join(folder, file_name)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5  # Timeout after 5 seconds
        )
        return result.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "Execution timed out"
    except subprocess.CalledProcessError as e:
        return None, e.output.strip()

def sanitize_id(item_id, dataset_name):
    return item_id.replace('/', '-').replace('_', '-')

def parse_args():
    parser = argparse.ArgumentParser(description="Process and Execute Generated Code Samples")
    parser.add_argument('--dataset_name',type=str, required=True, help="dataset_name")
    parser.add_argument('--json_file_path', type=str, required=True, help="Path to the JSON file containing the results to be processed.")
    parser.add_argument('--python_files_folder', type=str, required=True, help="Folder where the Python files are saved.")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the output results.")
    return parser.parse_args()


def main(args):
    data = read_json(args.json_file_path)
    total_samples = len(data)
    executable_count = 0
    correct_predictions = 0
    output_results = []
    predictions = []
    
    # Ensure the directory for saving sample Python files exists
    samples_code_folder = os.path.join(args.python_files_folder, args.output_folder)
    os.makedirs(samples_code_folder, exist_ok=True)

    for item in data:
        item_id = sanitize_id(item['id'], args.dataset_name) 
        file_name = f'question{item_id}.py'
        code = item['prediction'].replace('\\n', '\n') if item['prediction'] is not None else "print('')"
        # Save each Python file in the specified samples_code_folder
        save_to_file(file_name, code, samples_code_folder)

        result, error = execute_python_file(file_name, samples_code_folder)
        ground_truth = item['gold']
        if result is not None:
            executable_count += 1
            output = f'Sample ID: {item_id}, Output: {result}, Ground Truth: {ground_truth}'
            if result == ground_truth:
                correct_predictions += 1
        else:
            output = f'Sample ID: {item_id}, Execution Error: {error}'
        output_results.append(output)
        result_single_line = result.replace('\n', ' ').strip() if result is not None else 'ERROR'
        predictions.append(result_single_line)


    # Save the execution results summary in the same output folder
    save_to_file('output_results.txt', '\n'.join(output_results), samples_code_folder)
    save_to_file('predictions.txt', '\n'.join(predictions), samples_code_folder)
    executable_rate = (executable_count / total_samples) * 100

    print(f'Executable Rate: {executable_rate}%')

if __name__ == '__main__':
    args = parse_args()
    main(args)
