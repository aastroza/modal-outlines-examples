import os
from modal import App, Secret, gpu, enter, build, method, Image

app = App(name="outlines-app")

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.39",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)

@app.cls(image=outlines_image, secrets=[Secret.from_dotenv()], gpu=gpu.A100(memory=80), timeout=300)
class Model:
    @build()
    @enter()
    def import_model(self):
        import outlines

        self.model = outlines.models.transformers(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda",
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
            },
        )

    @method()
    def generate(self, schema: str, prompt: str):
        import outlines

        generator = outlines.generate.json(self.model, schema.strip())

        result = generator(prompt)

        return result