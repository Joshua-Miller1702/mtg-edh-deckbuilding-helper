Initial idea: 
- Hypergeometric calculator to be applied to groups of cards classified by an NLP ~~with a GUI for deck importing and odd checking~~.

Why: 
- I am manually classifying cards each time i build a new deck (one per week) which takes ~ 1 - 2h to do each time so this project allows me to do 2 things; 1. Refine and build decks much faster, 2. learn how to make machine learning models.

Expected result: 
- A funcitonal NLP model that has a decently high accuracy 90+% that can do the majority of deck labelling for me after which i can tweak manually or refine the model to make it more effective. 

1st Week: 
- Created hypergeometric calculator to calucalte exact draw probability of x cards from a group with user input. 
- Added function to grab and format MTGA formatted decklists from clipboard.
- Planned NLP data gathering.

2nd Week:
- Added random card grabber function that pulls cards from scryfall api for manual sorting.
- Added manual card sorter function and improved it to handle missinputs. Added ability for this function to generate csv files.
- Added functionality to hypergeometric calc to allow it to dispay x or less draws as this better reflects the usecase.
- Added card text retrival and text preprocessing functions to nlp_logistic_regression.
- Created initial model.
- Re-evaluated model as it had a flawed implementation.

3rd Week:
- Removed non-functional model.
- Redefined project goals.
- Began creation of new model.
- Fixed several issues with functions and defaulting.
- Created "functional" model.
- Generated larger dataset ~300 samples.
- Identified issues with multinomial logisitic regression and the dataset causing it to be a much more difficult model to implement than intended.
- Decided to shift to using a naieve bayes classifier instead, with option of circling back to the logistic regression model for fun.
- Re-organised folders to make more sense.
- Locally downloaded card dataset to stop hammering the scryfall api.

4th Week:
- Reevaluated approach, identified a knowladge gap in the nomanclature leading me down a rabbit hole making incorrect models (of which i made 2), not entirely a waste of time as they greatly benefited the learning process and in terms of how both models work and how my data is speicifically structured.
- Made a much larger dataset ~1000 more samples (totalling 1300 though this is still somewhat too low)
- Produced a functional one vs rest model using sklearn that is on ~90% accurate though 'true' accuracy is limited by the small dataset that results in the accuracy of some categories being 1.0 due to only having 1 value in the validation dataset.
- Scaled back project scope, while a GUI would be nice i do ultimately have to make this functional and learning to make a GUI is not why i started this project.

