import argparse
from typing import Dict, List

from transformers import pipeline


MODEL = "Domino-ai/distilbert-base-cased-wikiann"
ner = pipeline(aggregation_strategy="simple", task="token-classification", model=MODEL)


def predict_spans(sentences: List[str]) -> List[Dict]:
    results = ner(sentences)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate your fine-tuned model")
    parser.add_argument(
        "--eval_path",
        help="Path to eval data text file. Assumed newline separated strings",
        required=False,
    )
    args = parser.parse_args()
    default_test_sentence = "The Microsoft spokesperson Ronald Ramer lives in New York"
    if not args.eval_path:
        print("No eval_path provided. Using default test data")
        sentences = [default_test_sentence]
    else:
        with open(args.eval_path, "r") as f:
            sentences = f.readlines()
    spanss = predict_spans(sentences)
    print("*" * 50)
    for sentence, spans in zip(sentences, spanss):
        print("Input:", sentence)
        print("Spans:")
        for span in spans:
            print(f"\tEntity: {span['entity_group']}, Word: {span['word']}, Confidence: {span['score']}")
        print("*" * 50)


if __name__ == "__main__":
    main()
