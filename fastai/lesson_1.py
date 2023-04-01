# Re-implementation of the lesson 1 notebook in fastai v1 in a python script
# that I can run from the command line. Small edits.
# Author: Jeremy Howard

from duckduckgo_search import ddg_images
from fastdownload import download_url
from fastai.vision.all import Image
from fastai.vision.all import *
from fastcore.all import *
import fire
from time import sleep


def search_images(term, max_images=30):
    """
    Searches duckduckgo for images of a given term and returns a list of urls
    """
    print(f"Searching for '{term}...'")
    return [item["image"] for item in ddg_images(term, max_results=max_images)]


def main(
    search_terms: tuple,
    max_images: int = 10,
    generate_images: bool = False,
    use_existing_model: bool = False,
):
    label_a, label_b = search_terms
    path = Path(f"{label_a}_or_{label_b}")

    if generate_images:
        generate_labelled_dataset(search_terms, path, max_images)

    verify(path)
    dls = get_dls(path)

    if use_existing_model:
        model = load_learner("models/fine_tuned_resnet18.pkl")
    else:
        model = learn(dls)

    predict(model, label_a)


def generate_labelled_dataset(search_terms: tuple, path: Path, max_images: int = 10):
    """
    Simple wrapper around search_images and download_images
    """
    for search_term in search_terms:
        dest = path / search_term
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f"{search_term} photo"))
        sleep(0.5)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f"{search_term} sun photo"))
        sleep(0.5)
        download_images(dest, urls=search_images(f"{search_term} shade photo"))
        sleep(0.5)
        resize_images(path / search_term, max_size=400, dest=path / search_term)


def verify(path: str) -> None:
    """
    Verify that the images in the given path are valid.
    Note: Functional programming approach (new to me)
    """
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} images that failed to open")


def learn(dls, save: bool = True):
    """
    Returns the pre-trained resnet18 fine-tuned on the given dataset
    """
    model = vision_learner(dls, resnet18, metrics=error_rate)
    model.fine_tune(3)
    if save:
        model.export("fine_tuned_model.pkl")
    print(f"Finished fine-tuning. Error rate: {model.validate()}")
    return model


def get_dls(path, bs=32, size=192):
    """
    Returns a DataLoaders object for the given path
    """
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(size, method="squish")],
    ).dataloaders(path, bs=bs)


def predict(model: object, search_term: str):
    """
    Predicts the label of the given image
    Returns the label and the probability of that label
    """
    dest = f"{search_term}.jpg"
    urls = search_images(f"chocolate {search_term} photos", max_images=1)
    download_url(urls[0], dest, show_progress=False)
    name, _, probs = model.predict(PILImage.create(dest))

    print(f"This is a: {name}.")
    print(f"Probability it's a {name}: {probs[0]:.4f}")


if __name__ == "__main__":
    fire.Fire(main)

# main(('cake', 'tree'), max_images=30, generate_images=False, use_existing_model = True)
