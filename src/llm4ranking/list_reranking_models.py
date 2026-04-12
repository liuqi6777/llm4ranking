from llm4ranking import list_reranking_models


def main():
    for approach in list_reranking_models():
        print(approach)


if __name__ == "__main__":
    main()
