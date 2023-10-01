import argparse
from typing import List

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, pipeline

from finetune import MODEL_DIR

model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def predict_sentiment(sentences: List[str]):
    results = nlp(sentences)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate your fine-tuned model")
    parser.add_argument(
        "--eval_path",
        help="Path to eval data text file. Assumed newline separated strings",
        required=False,
    )
    args = parser.parse_args()
    default_test_sentence = "I can't stand how small the product is. The description made it seem much larger."
    if not args.eval_path:
        print("No eval_path provided. Using default test data")
        sentences = [default_test_sentence]
    else:
        with open(args.eval_path, "r") as f:
            sentences = f.readlines()
    sentiments = predict_sentiment(sentences)
    print("*" * 50)
    for sentence, sentiment in zip(sentences, sentiments):
        print("sentence        : {}".format(sentence))
        print("prediction      : {}".format(sentiment["label"]))
        print("confidence      : {}".format(sentiment["score"]))
        print("*" * 50)


if __name__ == "__main__":
    main()
