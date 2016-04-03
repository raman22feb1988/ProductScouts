### Problem Statement

Indix deals with product data. Most of our data source is the web. We collect information from ecommerce portals, parse them and add it to our index. One of the challenges we face with product data is to identify the brand a particular belongs to.

This episode of hackathon is going to expose you to the challenges in this space. You are given a product dataset which contains just 3 fields, product_title, brand_id and category_id (in order). The problem is to identify the brand_id, using the other features (product_title and category_id). You could treat this as a standard classification problem and arrive at the label (brand_id) for a given input record. The test set would have 2 fields - product_title and category_id.

We would use accuracy measurement to evaluate your classifier's performance.


### Tech Dependencies

`anaconda`

### How to Execute the training and testing

* `brand_classifier_combined.py` is the main source to be executed
*  Please place both the input file with names as below in same folder as the script
*  Training File : `classification_train.tsv`
*  Testing File : `classification_blind_set_corrected.tsv`
*  Run `python blind_classifier_combined.py`
*  The file `oddCategoryFinder.R` can be run R Studio to identify the best fit for each category. To be used in the rules for featuring.s
*  Once the script is executed output files are created in same folder in `.txt` format with file name of the format `output_<timestamp>.txt`



