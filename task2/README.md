# Task 2:  Named entity recognition + image classification

In this task, you will work on building your ML pipeline that consists of 2 models responsible for
totally different tasks. The main goal is to understand what the user is asking (NLP) and check if
he is correct or not (Computer Vision).

You will need to:
- find or collect an animal classification/detection dataset that contains at least 10
classes of animals.
- train NER model for extracting animal titles from the text. Please use some
transformer-based model (not LLM).
- Train the animal classification model on your dataset.
- Build a pipeline that takes as inputs the text message and the image.


# Solution details

## NER part

The main reason for using a transformer in this task is to generalize synonyms not included in the dataset and to disregard indirect mentions based on context. From this perspective, BIO tagging is both excessive and restrictive. Our goal is to understand what was mentioned, not where, so using I-[KIND] tagging is a better fit. This approach handles synonyms and plurals effectively and improves generalization.

The biggest problem of this part was a dataset. Details of dataset creation can be found in the [notebook](./ner_eda.ipynb).

At the end, I obtained three datasets:
- Labeled by LLM and filtered animal-fun-facts (~600 examples)
- Synthetic (~1,900 examples)
- Labeled manually MS COCO (~150 examples): one filtered from indirect animal mention (such as 'painted in zebra') and one not filtered.

Test on synthetic+facts test split:
```
'test_precision': 0.9198552223371251, 
'test_recall': 0.918904958677686, 
'test_f1': 0.9193798449612404, 
```
Test on coco_filtered:
```
'test_coco_filtered_precision': 0.9177489177489178, 
'test_coco_filtered_recall': 0.905982905982906, 
'test_coco_filtered_f1': 0.9118279569892475, 
```

Although 2,500 examples (mostly synthetic) are clearly insufficient for a production-grade model, it performs well on simple examples and even generalizes to synonyms not present in my datasets — likely due to the excellent pretraining of `microsoft/deberta-v3-base`.

Tests on set with indirect mentions showed poor (`'test_coco_precision': 0.7444, 'test_coco_recall': 0.8434, 'test_coco_f1': 0.7908`) performance, which is not surprising since there were almost no negative examples in the dataset. However, handling indirect mentions was not required for the task.

Example of generalization:
```
$ python3 ner_inference.py --verbose --input="bowwow"
['bow', 'wow']
['I-O', 'I-O']
[]

$ python3 ner_inference.py --verbose --input="bowwow was always the first to greet everyone at the door with a wagging tail" 
['bow', 'wow', 'was', 'always', 'the', 'first', 'to', 'greet', 'everyone', 'at', 'the', 'door', 'with', 'a', 'wagging', 'tail']
['I-DOG', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O', 'I-O']
dog
```

I added an option in the inference script and pipeline to detect multiple animal titles at the NER stage and mark them as *True* or *False*.

[Link to model weights.](https://huggingface.co/andriibul/animals-ner)


## Image classification part


I was pressed for time, so I decided not to experiment with different CNN architectures and chose ViT. A transformer is somewhat overkill for this task, but it was really easy to fine-tune thanks to another great pretrain (`google/vit-base-patch16-224-in21k`).

I took the Animal-10 dataset and downsampled it to address class imbalance.

My fine-tuned model is clearly missing some contrasting examples, as it classifies images like [this](assets/not_a_cow.jpg) or [this](assets/not_a_cat.jpg) as cows or cats. Overall, the transformer handled the task easily and performed well, achieving an F1-score of 0.98, which is not surprising.

[Link to model weights.](https://huggingface.co/andriibul/animal-imgcls)

# Setup
Go to `requirements.txt` and uncomment neccessery lines for GPU support if needed.
You can use any virtual environment you prefer.  
First, install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you want to use MLflow dashboards:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Train
Training scripts are parametrized with yaml configs. Fill in the parameters in `ner_train.yaml` and `imgcls_train.yaml`. Then, start training:
```bash
python3 ner_train.py
```
or
```bash
python3 imgcls_train.py
```

## Inference
As mentioned earlier, the model weights are stored on the Hugging Face hub. For inference:
```bash
python3 ner_inference.py --input="where is the little kittycat?" --verbose
```
```bash
python3 imgcls_inference.py --input="assets/cow.jpg"
```
To invoke pipeline:
```bash
python pipeline.py --ner_input="does this picture contain jumbo or cattle?" --ner_multiple --ner_verbose --imgcls_input="assets/not_a_cow.jpg" 
['does', 'this', 'picture', 'contain', 'jumbo', 'or', 'cattle', '?']
['I-O', 'I-O', 'I-O', 'I-O', 'I-ELEPHANT', 'I-O', 'I-COW', 'I-O']
{'elephant': False, 'cow': True}
```