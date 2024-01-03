from cog import BasePredictor, Input, ConcatenateIterator

class Predictor(BasePredictor):
    def setup(self):
        pass
    def predict(self, prompt: str = Input(description="Input prompt to generate from.")
    ) -> str:
        return "Hello world!"


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict("Hello world!")
    print(out)