#!/usr/bin/env python3
"""
QA bot using pretrained bert with local documents
as dictionary.
"""

def main():
    """
    Main question/answer loop.
    """
    question = ""
    quit_str = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input("Q: ")
        if question.lower() in quit_str:
            break
        print("A: ")

    print("A: Goodbye")

if __name__ == '__main__':
    main()
