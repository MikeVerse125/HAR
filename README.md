The primary task is to recognize human actions in still images. The model is designed to predict one of 40 possible actions depicted in the image and determine whether there is more than one person present. This implementation begins with training a baseline model and further enhances performance using channel attention mechanisms. Parameters can be customized by modifying the `config.yaml` file.

Run code:
    python main.py config.yaml

Run the prediction of the model on unlabel images:
    step 1: 
        Update the config file [type, num_of_predimg, model_path]
        Type of the model should match the model_path
    step 2: 
        Run code: python prediction.py config.yaml
    step 3: 
        Check the result in "results/predictions/prediction_result.png"