from llm4ranking import list_available_reranking_approaches


def main():
    for approach in list_available_reranking_approaches():
        print(approach)


if __name__ == "__main__":
    main()
