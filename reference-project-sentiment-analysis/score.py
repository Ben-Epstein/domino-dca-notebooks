from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import pipeline
import argparse
from finetune import MODEL_DIR
from typing import List

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def predict_sentiment(sentences: List[str] ):
    results = nlp(sentences)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate your fine-tuned model"
    )
    parser.add_argument(
        "--eval_path", help="Path to eval data text file. Assumed newline separated strings", required=False
    )
    args = parser.parse_args()
    default_test_sentence = "I can't stand how small the product is. The description made it seem much larger."
    if not args.eval_path:
        print("No eval_path provided. Using default test data")
        sentences = [default_test_sentence]
    else:
        with open(args.eval_path, "r") as f:
            sentences = f.readlines()
    sentiment = predict_sentiment(sentences)
    # print("sentence   : {}".format(test_sentence))
    # print("prediction : {}".format(sentiment["label"]))
    # print("score      : {}".format(sentiment["score"]))

if __name__ == "__main__":
    main()
