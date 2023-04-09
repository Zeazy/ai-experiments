from duckduckgo_search import ddg_images
from fastdownload import download_url
from pathlib import Path
from fastai.vision.all import Image
from fastai.vision.all import *
from fastcore.all import *
from time import sleep
from typing import List


def search_images(term, max_images=30):
    """
    Searches duckduckgo for images of a given term and returns a list of urls
    """
    print(f"Searching for '{term}...'")
    return [item["image"] for item in ddg_images(term, max_results=max_images)]


def main(
    search_terms: List[str],
    max_images: int = 10,
    generate_images: bool = False,
    use_existing_model: bool = False,
):
    current_dir = Path.cwd()
    path = current_dir / "images"

    if generate_images:
        for search_term in search_terms:
            generate_labelled_dataset(search_term, path / search_term, max_images)

    verify(path)
    dls = get_dls(path)

    if use_existing_model:
        model = load_learner(Path.cwd() / "/models/fine_tuned_resnet18.pkl")
    else:
        model = learn(dls)

    predict_from_path(model, search_terms[0])


def generate_labelled_dataset(search_term: str, path: Path, max_images: int = 10):
    """
    Simple wrapper around search_images and download_images
    """
    path.mkdir(exist_ok=True, parents=True)
    download_images(path, urls=search_images(f"{search_term} photo"))
    sleep(0.5)  # Pause between searches to avoid over-loading server
    download_images(path, urls=search_images(f"{search_term} in jungle photo"))
    sleep(0.5)
    download_images(path, urls=search_images(f"{search_term} in city photo"))
    sleep(0.5)
    resize_images(path, max_size=400, dest=path)


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
    model = vision_learner(dls, resnet34, metrics=error_rate)
    model.fine_tune(5)
    if save:
        print(Path.cwd() / "models/fine_tuned_resnet18_model.pkl")
        model.export(Path.cwd() / "models/fine_tuned_resnet18_model.pkl")
    print(f"Finished fine-tuning. Error rate: {model.validate()}")
    return model

def get_dls(path, bs=16, size=192):
    """
    Returns a DataLoaders object for the given path
    """
    images = get_image_files(path)
    print(images)
    
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(size, method="squish")],
    ).dataloaders(path, bs=bs, batch_tfms=aug_transforms())


def predict_from_path(model: object, image: str):
    """
    Predicts the label of the given image
    Returns the label and the probability of that label
    """
    dest = Path.cwd() / "images" / f"{image}"
    # pick random image from the given path
    dest = random.choice([x for x in dest.iterdir() if x.is_file()])
    print(dest)
    name, _, probs = model.predict(PILImage.create(dest))
    print(f"This is a: {name}.")
    print(f"Probability it's a {name}: {probs[0]:.4f}")


def predict(model: object, image: Image):
    pred_class, _ , pred_probs = model.predict(image)
    class_names = model.dls.vocab
    class_probs = list(zip(class_names, [round(float(prob.item()),5) for prob in pred_probs]))
    
    return pred_class, class_probs
