# Local Test

## Setup

```bash
conda activate llm-project
pip install -r requirements.txt
```

Set your API key:
```bash
export API_KEY="your-api-key-here"
```

## Run

```bash
python mas_loop.py --paper data/md/example_paper.md --n_iter 1
```

Optionally save output to a file:
```bash
python mas_loop.py --paper data/md/example_paper.md --n_iter 1 --output results.txt
```

---

# Webapp

## Setup

```bash
pip install flask
```

## Run

```bash
python webapp/app.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser.



# Dev note:
## Current workflow:
1. File upload
    - Missing: Ask user to upload file
    - Current: Manually upload pdf to `data/pdf`
2. Document preprocessing
    - Missing: Process user uploaded file
    - Current: Run `doc_preprocess.py`, output md file in `data/md`
3. Data segmentation
    - Current: 
        - Prompt user for file name
        - Display current markdown sections
        - Prompt user for corrected header level
        - Trigger `modular_seg.py` to output json file of raw and user-reviewed section-content dictionaries in `data/test`

## A better way to use OpenReviewer
1. Open Google Colab with G4 GPU running time. In terminal:
```bash
git clone https://huggingface.co/spaces/maxidl/openreviewer
cd openreviewer
pip install -r requirements.txt
pip install spaces gradio huggingface_hub
```
2. Double click app.py to open file and modify the 256th line as:
```python
demo.launch(share=True)
```
3. Return to terminal and run:
```bash
python app.py
```
4. Copy the public url in your local brower and use as usual. (Keep colab terminal running when using the public url)
