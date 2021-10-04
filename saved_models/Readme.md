The new model file located in the latest_ggnn_code new_model directory

NOTICE:
When Testing for Gated_PU80_r08 models(Gated_boost, Gated_noboost, GraphSage_boost, GraphSage_no_boost)\
please uncomment the self.before_mp, self.num_layers and self.batchnorm in model file\
Since the model was saved before I comment those out.

When testing for boost algorithm. Please leave the line 77 in the model file\
When testing for noboost algorithm. Please comment the line 77 in the model file

