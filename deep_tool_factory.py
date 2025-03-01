import json
import os
import argparse
import logging
from typing import List, Dict
from utils import OpenAIModel
from tqdm import tqdm
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_tool_creation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepSeekToolCreation:
    """Main class for handling tool creation using DeepSeek models."""
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._validate_paths()
        
        self.deepseek_api = OpenAIModel(
            api_key=args.api_key,
            api_base=args.api_base,
            model_name=args.model_name,
            stop_words=args.stop_words,
            max_new_tokens=args.max_new_tokens
        )

    def _validate_paths(self) -> None:
        if not os.path.exists(self.args.data_path):
            raise FileNotFoundError(f"Data path not found: {self.args.data_path}")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

    def load_dataset(self) -> List[Dict]:
        dataset_path = os.path.join(
            self.args.data_path, 
            self.args.dataset, 
            f'{self.args.split}.json'
        )
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {dataset_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        logger.info(f"Loaded {len(data)} samples from {self.args.dataset}")
        return data[:self.args.num_eval_samples] if self.args.num_eval_samples > 0 else data

    def construct_prompt(self, sample: Dict) -> str:
        prompt_path = os.path.join('prompts', f"prompt_{sample['dataset']}.txt")
        
        try:
            with open(prompt_path, 'r') as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {prompt_path}")
            raise

        return template.replace('[[QUESTION]]', sample['question'])

    @staticmethod
    def extract_python_code(response: str) -> str:
        code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
        return code_match.group(1).strip() if code_match else ''

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        try:
            prompts = [self.construct_prompt(sample) for sample in batch]
            responses = self.deepseek_api.batch_generate(prompts)
            
            return [{
                'id': sample['id'],
                'question': sample['question'],
                'gold': sample['answer'],
                'prediction': self.extract_python_code(response)
            } for sample, response in zip(batch, responses)]
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

    def run_inference(self, batch_size: int = 10) -> None:
        dataset = self.load_dataset()
        results = []
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset[i:i+batch_size]
            processed = self.process_batch(batch)
            results.extend(processed)
            logger.debug(f"Processed batch {i//batch_size + 1}/{len(dataset)//batch_size + 1}")

        self._save_results(results)

    def _save_results(self, results: List[Dict]) -> None:
        output_path = os.path.join(
            self.args.save_path,
            "answer.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSeek Tool Creation System")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument('--api_key', type=str, required=True,
                        help="DeepSeek API access key")
    parser.add_argument('--api_base', type=str, required=True,
                        help="DeepSeek API base")
    parser.add_argument('--model_name', type=str, default='deepseek/deepseek-chat',
                        help="Model identifier for DeepSeek")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Name of dataset to process")
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test', 'validation'],
                        help="Dataset split to use")
    parser.add_argument('--num_eval_samples', type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument('--stop_words', nargs='+', default=['------'],
                        help="Stop sequences for generation")
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Output directory for results")
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_arguments()
        tool_creator = DeepSeekToolCreation(args)
        tool_creator.run_inference(batch_size=10)
    except Exception as e:
        logger.error(f"Critical error occurred: {e}", exc_info=True)
        raise