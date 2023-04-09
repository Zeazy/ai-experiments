# 🦏 Rhino Predictor Web App 🦏

This web app is part of the FastAI course and the main objective was to apply what I've learned to fine-tune a Resnet34 model to predict rhino species.

## Description

The Rhino Predictor web app is a simple image classification app that generates a random picture of a rhino and asks users to guess the species. The app uses a small dataset of rhino images to fine-tune a pre-trained Resnet34 model and classify the species of the rhino in the image.

The user interface is built using Streamlit, a Python library for building web apps, and the model is built using the FastAI library.

## Installation

To run the Rhino Predictor web app, follow these steps:

1. Clone the repository: `git clone https://github.com/Zeazy/ai-experiments.git`
2. Navigate to the repository: `cd experiments/image_predictor`
3. Install the required libraries: `pip install -r requirements.txt`
4. Run the web app: `streamlit run app.py`

You can also find the hosted web app on Hugging Face [here](https://huggingface.co/spaces/Zeazy/rhino_identifier).

## Usage

Once the web app is running, click on the "Search for a rhino" button to generate a random picture of a rhino. Then select the species of the rhino from the radio buttons and click on "Submit" to make your guess. The app will display whether your guess was correct or not, along with the probabilities for each species.

## Contributing

Contributions are welcome! If you find any bugs or issues with the web app, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0/).
