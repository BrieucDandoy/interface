from uuid import uuid4
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
logging.getLogger('transformers').setLevel(logging.WARNING)


root_logger = logging.getLogger()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = "main.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



class prompt_manager:
    def __init__(self,path : str ,name = None) -> None:
        if name is None:
            self.prompt_id = uuid4()
        else:
            self.prompt_id = name
        self.history = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16).to(self.device)
        self.model = self.model.eval()
        self.model.cuda()
    def save_prompt(self):
        with open(f'{self.prompt_id}.json','w') as f:
            json.dump(self.history, f)

    def rename(self,name):
        self.prompt_id = name

    def get_response(self,user_input):
        if not self.history:
            self.history = [
    {"role": "system", "content": "You are a friendly chatbot who always kindly"},
]

        self.history.append({"role": "user", "content": user_input})
        model_input = self.tokenizer.apply_chat_template(self.history,add_generation_prompt=True,return_tensors='pt').to(self.device)
        input_length = model_input.shape[1]
        outputs = self.model.generate(
        model_input,
        max_new_tokens=1024,  # Maximum length of the generated text
        num_return_sequences=1,  # Number of different sequences to generate
        temperature=0.5,  # Controls randomness in sampling
        top_k=100,  # No filtering based on token probabilities
        top_p = 0.95,
        repetition_penalty = 2.0,
        use_cache= True,
        do_sample=True)
        decoded_output = self.tokenizer.batch_decode(outputs[:,input_length:], skip_special_tokens=True)[0]
        self.history.append({"role": "system", "content": decoded_output})
        return decoded_output

class input_manager:
    def __init__(self,path : str) -> None:
        self.path = path
        self.current_prompt = prompt_manager(path)
        self.liste_prompt = [self.current_prompt]
        self.run = True
    def get_input_type(self, user_input : str) -> str:
        if user_input[0] == ':':
            return 'command'
        else:
            return 'prompt'
        

    def get(self,user_input):
        intput_type = self.get_input_type(user_input)
        if intput_type == 'command':
            self.execute_command(user_input)
            return None
        elif intput_type == 'prompt':
            return self.get_model_output(user_input)
    

    def execute_command(self,command):
        if command == ':q':
            self.run = False
        elif command == ':w':
            self.save_prompt()
        elif command == 'wq':
            self.save_prompt()
            self.run = False

        elif command[:7] == ':rename':
            new_name = command[8:]
            self.current_prompt.rename(new_name)

        elif command[:3] == 'new':
            name = command[3:]
            if name == '':
                self.add_prompt()
            else:
                self.add_prompt(name)


    def get_model_output(self,user_input):
        return self.current_prompt.get_response(user_input)

    def add_prompt(self,name = None):
        new_prompt = prompt_manager(name)
        self.liste_prompt.append(new_prompt)
        self.current_prompt = new_prompt

    def __bool__(self):
        return self.run