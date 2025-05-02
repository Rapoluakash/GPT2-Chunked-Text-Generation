from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2Model,
    TFGPT2Model,
    pipeline,
    set_seed
)
import torch

# Load the GPT-2 tokenizer and model for generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 has no pad_token by default, so set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def chunk_text(text, max_length=512):
    """Chunk text into smaller parts."""
    tokens = tokenizer.encode(text, return_tensors='pt')[0]
    chunks = []

    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)

    return chunks

def generate_responses(chunks, max_new_tokens=100):
    """Generate a response for each chunk using the LLM."""
    responses = []

    for i, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0)  # Add batch dimension
        attention_mask = torch.ones_like(input_ids)

        try:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
        except Exception as e:
            responses.append(f"[Error generating chunk {i+1}: {str(e)}]")

    return responses

# Simulated long input text
long_text = "Indian cricket is followed passionately. " * 50

# Process and generate
chunks = chunk_text(long_text)
responses = generate_responses(chunks)

# Output
for i, response in enumerate(responses):
    print(f"\n🧩 Response for chunk {i+1}:\n{response}\n{'-'*60}")

# Additional: Show tensor outputs from GPT2Model
print("\n--- Hidden State from GPT2Model ---")
tokenizer_pytorch = GPT2Tokenizer.from_pretrained("gpt2")
model_pytorch = GPT2Model.from_pretrained("gpt2")
text = "Replace me by any text you'd like."
encoded_input = tokenizer_pytorch(text, return_tensors='pt')
output = model_pytorch(**encoded_input)
print(output)

# Additional: TensorFlow variant (only works if TensorFlow is installed)
try:
    tokenizer_tf = GPT2Tokenizer.from_pretrained("gpt2")
    model_tf = TFGPT2Model.from_pretrained("gpt2")
    encoded_input_tf = tokenizer_tf(text, return_tensors='tf')
    output_tf = model_tf(encoded_input_tf)
    print("\n--- Hidden State from TFGPT2Model ---")
    print(output_tf)
except Exception as e:
    print("\n[TensorFlow GPT-2 skipped: TensorFlow not installed or GPU unavailable]")
    print(str(e))

# Bonus: Test sampling with pipeline
print("\n--- Sampling Test ---")
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
outputs = generator("The Black man worked as a", max_length=10, num_return_sequences=3)
for o in outputs:
    print(o['generated_text'])
