import inspect

from src.model.model_setup import load_model


def main():

    _, model = load_model()

    print("\n--- MODEL FORWARD SIGNATURE ---")

    signature = inspect.signature(
        model.forward
    )

    print(signature)

    print("\n--- MODEL TYPE ---")
    print(type(model))


if __name__ == "__main__":
    main()