from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig


huggingface_dataset_name = "knkarthick/dialogsum"

# Load the dataset using Hugging Face's datasets library
dataset = load_dataset(huggingface_dataset_name)


example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()
    
    
# Specify the model name
model_name='google/flan-t5-base'

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


 # Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)



# Define a test sentence
sentence = "This is a test sentence."

# Encode the sentence using the tokenizer, returning PyTorch tensors
sentence_encoded = tokenizer(sentence, return_tensors='pt')

# Decode the encoded sentence, skipping special tokens
sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0], 
        skip_special_tokens=True
    )

# Print the encoded sentence's representation
print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])

# Print the decoded sentence
print('\nDECODED SENTENCE:')
print(sentence_decoded)



# Iterate through example indices, where each index represents a specific example
for i, index in enumerate(example_indices):

    # Retrieve dialogue and summary for the current example
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    # Tokenize the dialogue and convert it to a vector of PyTorch tensors
    inputs = tokenizer(dialogue, return_tensors='pt')

    # Generate an output using the model, limiting the new tokens to 50
    # This uses the LLM to generate a summary of the dialogue without any prompt engineering
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )

    # Show the results
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')
    
    
    
    
# Iterate through example indices, where each index represents a specific example
for i, index in enumerate(example_indices):
    # Retrieve dialogue and summary for the current example
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    # Construct an instruction prompt for summarizing the dialogue 
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
    """

    # Tokenize the constructed prompt and convert it to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate an output using the model, limiting the new tokens to 50
    # This uses the LLM to generate a summary of the dialogue with the constructed prompt
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    # Show the results
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)    
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
    
    


# Iterate through example indices, where each index represents a specific example
for i, index in enumerate(example_indices):
    # Retrieve dialogue and summary for the current example
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    # Construct a prompt for summarizing the dialogue using the FLAN-T5 template
    prompt = f"""
Dialogue:

{dialogue}

What was going on?
"""

    # Tokenize the constructed prompt and convert it to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate an output using the model, limiting the new tokens to 50
    # This uses the LLM to generate a summary of the dialogue with the constructed prompt
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    # Show the results
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
    
    
    
    def make_prompt(full_examples_indices, index_to_summarize):
    """
    Construct a prompt for one-shot or few-shot inference.

    Parameters
    ----------
    full_examples_indices : list
        A list containing indices for complete dialogues to be included in the prompt. These dialogues serve as examples 
        for the model to learn from (for one-shot or few-shot inference).
    index_to_summarize : int
        The index for the dialogue that the model is expected to give a summary for.

    Returns
    -------
    str
        A prompt string that is constructed as per the given parameters - full dialogues examples followed by a dialogue 
        that needs to be summarized.
    """
    prompt = ''

    # Go through each index in the full examples list
    for index in full_examples_indices:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # Add each dialogue and its summary to the prompt string, followed by a stop sequence. The stop sequence 
        # '{summary}\n\n\n' is essential for FLAN-T5 model. Other models may have their own different stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    # Now add the dialogue that needs to be summarized by the model
    dialogue_to_summarize = dataset['test'][index_to_summarize]['dialogue']

    # Append this new dialogue to the prompt string
    prompt += f"""
Dialogue:

{dialogue_to_summarize}

What was going on?
"""

    # Return the constructed prompt
    return prompt



 # Define index for full example to be included in the prompt as a one-shot example
full_examples_indices = [40]
# Define the index for the dialogue that the model is expected to give a summary for
example_index_to_summarize = 200

# Create the prompt for one-shot inference
one_shot_prompt = make_prompt(full_examples_indices, example_index_to_summarize)

print(one_shot_prompt)



# Retrieve the human-generated summary for the 'example_index_to_summarize' example
summary = dataset['test'][example_index_to_summarize]['summary']

# Tokenize the one-shot prompt and convert it to PyTorch tensors
inputs = tokenizer(one_shot_prompt, return_tensors='pt')

# Generate an output using the model, limiting the new tokens to 50
# This uses the LLM to generate a summary of the dialogue with the one-shot prompt
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

# Show the results
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')



# Define indices for full examples to be included in the prompt as a few-shot examples 
full_examples_indices = [40, 80, 120]
# Define the index for the dialogue that the model is expected to give a summary for
example_index_to_summarize = 200

# Create the prompt for few-shot inference
few_shot_prompt = make_prompt(full_examples_indices, example_index_to_summarize)

print(few_shot_prompt)




# Retrieve the human-generated summary for the specified example
summary = dataset['test'][example_index_to_summarize]['summary']

# Tokenize the few-shot prompt and convert it to PyTorch tensors
inputs = tokenizer(few_shot_prompt, return_tensors='pt')

# Generate an output using the model, limiting the new tokens to 50
# This uses the LLM to generate a summary of the dialogue with the few-shot prompt
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

# Show the results
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')



# Define a GenerationConfig with specific parameters
generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)

# Tokenize the few-shot prompt and convert it to PyTorch tensors
inputs = tokenizer(few_shot_prompt, return_tensors='pt')

# Generate an output using the model, limiting the new tokens to 50
# This uses the LLM to generate a summary of the dialogue with the few-shot prompt
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0], 
    skip_special_tokens=True
)

# Show the results
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')




