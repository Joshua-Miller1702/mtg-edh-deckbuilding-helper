Initial idea: Hypergeometric calculator to be applied to groups of cards classified by an NLP.

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
- Generated larger dataset.
- Identified issues with multinomial logisitic regression and the dataset causing it to be a much more difficult model to implement than intended.
- Decided to shift to using a naieve bayes classifier instead, with option of circling back to the logistic regression model for fun.
- Re-organised folders to make more sense.

