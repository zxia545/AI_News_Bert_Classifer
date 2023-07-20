from transformers import BertTokenizer, BertModel

model_path = '/home/data/zhengyuhu/bert/bert-base-uncased'

# ------- BertTokenizer ------
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text,padding='max_length', 
                       max_length = 10, 
                       truncation=True,
                       return_tensors="pt")
# ------- bert_input ------
print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])

#--------- bert_output -----------
example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)

output = model(**bert_input)
print(output)