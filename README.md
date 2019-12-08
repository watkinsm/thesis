## Intent Detection: Development of the First Component for a Natural Language User Interface

#### Abstract
When developing systems which transform a user query into some kind of output, the first important step in the process is to determine the user's intent. By understanding what the user sought to accomplish with their text input, we are able to map their utterance to an appropriate function, which will perform the desired task, and report back on the results. In this thesis, we explore some current methods in intent detection, which we use as a foundation to develop our own intent detection component. This component will be used in a future natural language user interface for a source code anaylsis company. We find that, when given enough data, fine-tuning a large pre-trained language model tends to work better than training a model from scratch. We also show moderately good results for cross-lingual transfer when fine-tuning a pre-trained multilingual model on data from one language and evaluating it on data from a second language.

#### Data
ATIS: https://github.com/yvchen/JointSLU <br>
SNIPS: https://github.com/snipsco/nlu-benchmark <br>
Almawave-SLU: (please contact Almawave directly for the dataset)