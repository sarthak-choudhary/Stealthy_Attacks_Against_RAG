from openai import AzureOpenAI
from .Model import Model
import time

class AzureGPT(Model):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys))

        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.temperature = config["params"].get("temperature", 0.1)
        self.name = config["params"]["model_name"]

        self.client = AzureOpenAI(
            api_key=api_keys[api_pos],
            api_version=config["api_key_info"]["api_version"],
            azure_endpoint=config["api_key_info"]["azure_endpoint"],
            azure_deployment=config["api_key_info"]["deployment_name"]
        )

    def query(self, msg, top_tokens=1000000):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )

            response = completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            response = ""
        
        if response is None:
            response = ""
        return response
    
    def mutate_query(self, sentence):
        system_msg = 'You are a helpful and creative assistant who writes well.'
        user_message = f'Please revise the following sentence with no changes to its length and only output the revised version, the sentence is: \n "{sentence}".'
        revised_sentence = sentence
        received = False

        while not received:
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_message}
                    ],
                )

                revised_sentence = completion.choices[0].message.content.replace('\n', '')
                received = True
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

        if revised_sentence.startswith("'") or revised_sentence.startswith('"'):
            revised_sentence = revised_sentence[1:]
        if revised_sentence.endswith("'") or revised_sentence.endswith('"'):
            revised_sentence = revised_sentence[:-1]
        if revised_sentence.endswith("'.") or revised_sentence.endswith('".'):
            revised_sentence = revised_sentence[:-2]
        return revised_sentence