# Advanced Deep Learning Final Project

## Task

Develop and compare two deep learning models ([LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)) forsentiment analysis on Yelp reviews to determine the overall sentiment (positive,negative, neutral) expressed within a review. Utilize a dataset of Yelp reviews,focusing on businesses like restaurants or hotels. These reviews often containsubjective opinions and sentiments toward various aspects of the service or product.Analyze and compare the strengths and weaknesses of LSTM and Transformermodels for sentiment analysis on this specific dataset. Train the LSTM andDistilBERT models on the pre-processed and labelled Yelp review data forsentiment classification. Evaluate the performance of both models on a held-out testset of Yelp reviews using metrics like accuracy, precision, recall, and F1 score.Explore hyperparameter tuning for both models to optimize their performance.Analyze the types of reviews where each model performs better (e.g., short vs. longreviews, specific vocabulary usage). Consider including visualization techniques tounderstand how each model interprets and classifies sentiment within the reviews.Conduct a comparative analysis to understand the strengths and weaknesses of eachmodel on this specific task

## Deliverables

- A zip file containing (no RAR, no 7z, etc.):
  - A professional report as a PDF file containing:
    - Introduction of the project
    - Description of the dataset, data preprocessing methods andclassification method.
    - Your solutions, findings, and results on training, validation andtesting datasets in tabular and graphical format.
    - Interpretation, discussion, and Conclusion
    - References of all pre-existing resources used in the project.
  - Python file(s) (*.py*, *.ipynb*) containing your codes and results and areadme file showing how to use your programs.
    - The results should be displayed as the outputs of the program'sexecutions in the *.ipynb file.
    - The source codes should have the necessary lines of comments.
- You must specify each group member's tasks during the project activities

## Task breakdown

### Data Preparation

- Acquire a dataset of Yelp reviews focusing on restaurants or hotels.
- Pre-process the data by cleaning, tokenizing, and converting text into numerical representations suitable for deep learning models.
- Label the reviews with sentiment categories (positive, negative, neutral).

### Model Development

- Implement an LSTM (Long Short-Term Memory) model architecture for sentiment analysis.
- Implement a DistilBERT (a distilled version of BERT) model architecture for sentiment analysis.
- Train both models using the pre-processed and labeled Yelp review data for sentiment classification.

### Model Evaluation

- Split the dataset into training, validation, and test sets.
- Train the LSTM and DistilBERT models on the training data.
- Evaluate the performance of both models on the held-out test set using metrics such as accuracy, precision, recall, and F1 score.
- Analyze the strengths and weaknesses of each model based on performance metrics.

### Hyperparameter Tuning

- Explore hyperparameter tuning techniques for both LSTM and DistilBERT models to optimize their performance.
- Experiment with different hyperparameters such as learning rate, batch size, number of layers, hidden units, etc.

### Analysis of Model Performance

- Analyze the types of reviews where each model performs better (e.g., short vs. long reviews, specific vocabulary usage).
- Utilize visualization techniques to understand how each model interprets and classifies sentiment within the reviews.
- Identify patterns or trends in model performance across different types of reviews.

### Comparative Analysis

- Conduct a comparative analysis to understand the strengths and weaknesses of LSTM and DistilBERT models on the specific task of sentiment analysis on Yelp reviews.
- Compare the computational efficiency, interpretability, and generalization capabilities of both models.
- Discuss the implications of the findings and potential real-world applications.

### Documentation and Reporting

- Document the entire process including data preparation, model development, evaluation, hyperparameter tuning, and analysis.
- Prepare a comprehensive report detailing the methodology, results, and conclusions of the study.
- Present the findings in a clear and organized manner, including tables, charts, and visualizations to support the analysis.
